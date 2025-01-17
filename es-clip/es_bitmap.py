#!/usr/bin/env python3

import os

# os.environ['OPENBLAS_NUM_THREADS'] = '20'   # set the number of threads used by OpenBLAS
# os.environ['OMP_NUM_THREADS'] = '20'        # set the number of threads used by OpenMP

import argparse
# import cProfile
import json
# import multiprocessing as mp
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mp = torch.multiprocessing.get_context('spawn') # fork for preserving all variables in the main process
    
import os
import re

# import numpy as np
from numpy import mean
# from PIL import Image
from pgpelib import PGPE

from utils import *
from painter import TrianglesPainter
from hooks import (PrintStepHook, PrintCostHook, SaveCostHook, StoreImageHook, ShowImageHook)

def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='es_bitmap_out')
    parser.add_argument('--height', type=int, default=200, help='Height of the canvas. -1 for inference.')
    parser.add_argument('--width', type=int, default=-1, help='Width of the canvas.  -1 for inference.')
    parser.add_argument('--target_fn', type=str, required=True)
    parser.add_argument('--n_triangle', type=int, default=50)
    parser.add_argument('--loss_type', type=str, default='l2')
    parser.add_argument('--alpha_scale', type=float, default=0.5)
    parser.add_argument('--coordinate_scale', type=float, default=1.0)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--n_population', type=int, default=256)
    parser.add_argument('--n_iterations', type=int, default=10000)
    # parser.add_argument('--mp_batch_size', type=int, default=1)
    parser.add_argument('--solver', type=str, default='pgpe', choices=['pgpe','ga']) # maybe add more solvers
    parser.add_argument('--report_interval', type=int, default=50)
    parser.add_argument('--step_report_interval', type=int, default=50)
    parser.add_argument('--save_as_gif_interval', type=int, default=50)
    # parser.add_argument('--profile', type=bool, default=False)  # for proformance profiling
    cmd_args = parser.parse_args()
    return cmd_args

def init_training():
    """Initialize training - create working directory, dump args
    """
    
    # Create working directory
    global args
    os.makedirs(args.out_dir, exist_ok=True)
    assert os.path.isdir(args.out_dir)
    prev_ids = [re.match(r'^\d+', fn) for fn in os.listdir(args.out_dir)]
    new_id = 1 + max([-1] + [int(id_.group()) if id_ else -1 for id_ in prev_ids])
    desc = f'{os.path.splitext(os.path.basename(args.target_fn))[0]}-' \
           f'{args.n_triangle}-triangles-' \
           f'{args.n_iterations}-iterations-' \
           f'{args.n_population}-population-' \
           f'{args.solver}-solver-' \
           f'{args.loss_type}-loss'
    args.working_dir = os.path.join(args.out_dir, f'{new_id:04d}-{desc}')

    os.makedirs(args.working_dir)
    args_dump_fn = os.path.join(args.working_dir, 'args.json')
    with open(args_dump_fn, 'w') as f:
        json.dump(vars(args), f, indent=4) # vars() Namespace -> dict

    # Infer height and width
    height, width = infer_height_and_width(args.height, args.width, args.target_fn)

    # Load target image
    global target_arr
    target_arr = load_target_as_tensor(args.target_fn, (height, width)).cuda()
    save_as_png(os.path.join(args.working_dir, 'target'), tensor2img(target_arr))

    # Create painter
    global painter
    painter = TrianglesPainter(
        h=height,
        w=width,
        n_triangle=args.n_triangle,
        alpha_scale=args.alpha_scale,
        coordinate_scale=args.coordinate_scale, # Default to 1, to scale the coordinate
    )
    
    global loss_type
    loss_type = args.loss_type

    # record log
    global hooks
    hooks = [
        (args.step_report_interval, PrintStepHook()),
        (args.report_interval, PrintCostHook()),
        (args.report_interval, SaveCostHook(save_fp=os.path.join(args.working_dir, 'cost.txt'))),
        (
            args.report_interval,
            StoreImageHook(
                render_fn=lambda params: painter.render(params, background='white'),
                save_fp=os.path.join(args.working_dir, 'animate-background=white'),
                fps=args.fps,
                save_interval=args.save_as_gif_interval,
            ),
        ),
        (args.report_interval, ShowImageHook(render_fn=lambda params: painter.render(params, background='white'))),
    ]

def reload_vars(painter_, loss_type_, target_arr_):
    global painter, loss_type, target_arr
    painter, loss_type, target_arr = painter_, loss_type_, target_arr_
    

def fitness_fn(params, NUM_ROLLOUTS = 5):
    """Calculate Fitness

    Args:
        params (_type_): one solution
        NUM_ROLLOUTS (int, optional): Number of rollouts. Defaults to 5.

    Returns:
        float: minus fitness
    """
    losses = []
    target_arr_rgb = target_arr[..., :3]
    target_arr_rgb = target_arr_rgb / 255.
    
    for _ in range(NUM_ROLLOUTS):
        rendered_arr = painter.render(params).cuda()
        rendered_arr_rgb = rendered_arr[..., :3]
        rendered_arr_rgb = rendered_arr_rgb / 255.
        
        if loss_type == 'l2':
            pixelwise_l2_loss = (rendered_arr_rgb - target_arr_rgb)**2
            l2_loss = pixelwise_l2_loss.mean()
            loss = l2_loss
        elif loss_type == 'l1':
            pixelwise_l1_loss = torch.abs(rendered_arr_rgb - target_arr_rgb)
            l1_loss = pixelwise_l1_loss.mean()
            loss = l1_loss
        else:
            raise ValueError(f'Unsupported loss type \'{loss_type}\'')
        losses.append(loss)

    return -torch.mean(torch.Tensor(losses))  # pgpe *maximizes*

def fitnesses_fn(solutions):
    """
    Calculates the fitness values for a list of solutions.

    Args:
        solutions (list): A list of solutions.

    Returns:
        list: A list of fitness values corresponding to each solution.
    """
    proc_pool = mp.Pool(initializer=reload_vars,initargs=(painter,loss_type,target_arr)) # Process Pool
    out = proc_pool.map(func=fitness_fn, iterable=solutions)
    proc_pool.close();proc_pool.join();
    return list(out)

def PGPE_train():
    pgpe_solver = PGPE(
        solution_length=painter.n_params,
        popsize=args.n_population,
        optimizer='clipup',
        optimizer_config={'max_speed': 0.15},
    )

    best_params_fn = lambda _ : pgpe_solver.center

    global hooks
    for i in range(1, 1 + args.n_iterations):
        solutions = pgpe_solver.ask() # get solutions
        fitnesses = fitnesses_fn(solutions)
        
        # tell solver the fitnesses
        pgpe_solver.tell(fitnesses)

        # call hooks, record logs
        for (trigger_itervel, hook_fn_or_obj) in hooks:
            if i % trigger_itervel == 0:
                hook_fn_or_obj(i = i, solver = pgpe_solver, fitnesses_fn = fitnesses_fn, best_params_fn=best_params_fn)

def GA_train():
    from sko.GA import GA
    ga_solver = GA(func=lambda sol: -fitness_fn(sol, 5), n_dim=painter.n_params, size_pop=args.n_population, max_iter=args.n_iterations, prob_mut=0.001, lb=[0]*painter.n_params, ub=[1]*painter.n_params, precision=1e-6)
    ga_solver.to(device=device)
    ga_solver.run()
    
    # Backtracking to hook
    for i in range(len(ga_solver.generation_best_X)):
        solution = ga_solver.generation_best_X[i]
        fitness = ga_solver.generation_best_Y[i]
        
        for (trigger_itervel, hook_fn_or_obj) in hooks:
            if (i+1) % trigger_itervel == 0:
                hook_fn_or_obj(i = i+1, solver = ga_solver, fitnesses_fn = lambda _ : fitness , best_params_fn=lambda _ : solution)
    # save_as_gif(f'test_es_bitmap-ITER{args.n_iterations}-POP{args.n_population}.gif', images, fps=100)

def main():
    global args, painter, target_arr, loss_type, hooks
    args = parse_cmd_args()
    init_training() # create working directory and dump args
    
    # if args.profile:
    #     cProfile.runctx('training_loop(args)', globals(), locals(), sort='cumulative')
    # else:
    match args.solver:
        case 'pgpe':
            PGPE_train()
        case 'ga':
            GA_train()
        case _:
            raise ValueError(f'Unsupported solver: {args.solver}')
    
if __name__ == "__main__":
    main()
