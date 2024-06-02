#!/usr/bin/env python3

import os

os.environ['OPENBLAS_NUM_THREADS'] = '20'   # set the number of threads used by OpenBLAS
os.environ['OMP_NUM_THREADS'] = '20'        # set the number of threads used by OpenMP

import argparse
# import cProfile
import json
# import multiprocessing as mp
import torch
if torch.cuda.is_available():
    torch.device('cuda')
mp = torch.multiprocessing.get_context('fork') # fork for preserving all variables in the main process
    
import os
import re

import numpy as np
# from PIL import Image
from pgpelib import PGPE

from utils import (img2arr, arr2img, rgba2rgb, save_as_png, EasyDict, load_target, infer_height_and_width)
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
    parser.add_argument('--mp_batch_size', type=int, default=1)
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
    target_arr = load_target(args.target_fn, (height, width))
    save_as_png(os.path.join(args.working_dir, 'target'), arr2img(target_arr))

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

def fitness_fn(params, NUM_ROLLOUTS = 5):
    """Calculate Fitness

    Args:
        params (_type_): one solution
        NUM_ROLLOUTS (int, optional): Number of rollouts. Defaults to 5.

    Returns:
        float: minus fitness
    """
    losses = []
    for _ in range(NUM_ROLLOUTS):
        rendered_arr = painter.render(params)
        rendered_arr_rgb = rendered_arr[..., :3]
        rendered_arr_rgb = rendered_arr_rgb.astype(np.float32) / 255.

        target_arr_rgb = target_arr[..., :3]
        target_arr_rgb = target_arr_rgb.astype(np.float32) / 255.

        if loss_type == 'l2':
            pixelwise_l2_loss = (rendered_arr_rgb - target_arr_rgb)**2
            l2_loss = pixelwise_l2_loss.mean()
            loss = l2_loss
        elif loss_type == 'l1':
            pixelwise_l1_loss = np.abs(rendered_arr_rgb - target_arr_rgb)
            l1_loss = pixelwise_l1_loss.mean()
            loss = l1_loss
        else:
            raise ValueError(f'Unsupported loss type \'{loss_type}\'')
        losses.append(loss)

    return np.mean(losses)  # pgpe *maximizes*

def fitnesses_fn(solutions):
    """
    Calculates the fitness values for a list of solutions.

    Args:
        solutions (list): A list of solutions.

    Returns:
        list: A list of fitness values corresponding to each solution.
    """
    return list(map(fitness_fn, solutions))

def batching_fitnesses_fn(solutions):
    """
    Calculate the fitnesses of a list of solutions in batches using multiprocessing.

    Args:
        solutions (list): A list of solutions to calculate fitnesses for.

    Returns:
        list: A list of fitness values corresponding to each solution.
    """
    proc_pool = mp.Pool(initargs=(painter,loss_type,target_arr)) # Process Pool
    batches_in = (solutions[start:start + args.mp_batch_size] for start in range(0, len(solutions), args.mp_batch_size)) # split solutions into batches
    batches_out = proc_pool.imap(func=fitnesses_fn, iterable=batches_in) # map fitness_fn to each batch
    fitnesses = [item for batch in batches_out for item in batch]
    
    proc_pool.close();proc_pool.join(); # close and join the pool
    return fitnesses

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
        fitnesses = -np.array(batching_fitnesses_fn(solutions))
        
        # tell solver the fitnesses
        pgpe_solver.tell(fitnesses)

        # call hooks, record logs
        for (trigger_itervel, hook_fn_or_obj) in hooks:
            # trigger_itervel, hook_fn_or_obj = hook
            if i % trigger_itervel == 0:
                hook_fn_or_obj(i = i, solver = pgpe_solver, fitnesses_fn = lambda solutions: -np.array(batching_fitnesses_fn(solutions)), best_params_fn=best_params_fn)

def GA_train():
    from sko.GA import GA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ga_solver.to(device=device)
    ga_solver = GA(func=lambda sol: fitness_fn(sol, 5), n_dim=painter.n_params, size_pop=args.n_population, max_iter=args.n_iterations, prob_mut=0.001, lb=[0]*painter.n_params, ub=[1]*painter.n_params, precision=1e-3)
    ga_solver.run()
    
    # Backtracking to hook
    i=0
    for solution in ga_solver.generation_best_X:
        i+=1
        for (trigger_itervel, hook_fn_or_obj) in hooks:
            if i % trigger_itervel == 0:
                hook_fn_or_obj(i = i, solver = ga_solver, fitnesses_fn = batching_fitnesses_fn, best_params_fn=lambda _ : solution)

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
