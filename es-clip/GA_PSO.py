def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

from painter import TrianglesPainter
from es_bitmap import load_target, EasyDict, PrintCostHook, PrintStepHook, SaveCostHook, StoreImageHook, ShowImageHook
import numpy as np
import os
from utils import arr2img
import matplotlib.pyplot as plt
height, width = 200, 200
global painter, target_arr, loss_type
painter = TrianglesPainter(h=200, w=200, n_triangle=50, alpha_scale=0.1, coordinate_scale=1.0)
target_arr = load_target('assets/monalisa.png', (height, width))
loss_type = 'l2'


def parse_args(cmd_args=None):
    args = EasyDict()
    out_dir = 'es_bitmap_out'
    fps = 12
    report_interval = 50
    step_report_interval = 50
    save_as_gif_interval = 50
    args.out_dir = out_dir
    args.fps = fps
    args.report_interval = report_interval
    args.step_report_interval = step_report_interval
    args.save_as_gif_interval = save_as_gif_interval
    args.working_dir = os.path.join(out_dir, 'working')
    return args

args = parse_args()

def fitness_fn(params):
    NUM_ROLLOUTS = 5
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

    return -np.mean(losses)  # pgpe *maximizes*

# PSO:
from sko.PSO import PSO
# c1: 个体学习因子(cognitive factor)
# c2: 社会学习因子(social factor)
# w: 惯性权重
# pso = PSO(func=fitness_fn, n_dim=500, pop=40, max_iter=1000, lb=[0]*500, ub=[1]*500, w=0.8, c1=0.5, c2=0.5)
# pso.run()
# print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

# GA:
from sko.GA import GA
ga = GA(func=fitness_fn, n_dim=500, size_pop=256, max_iter=100, prob_mut=0.001, lb=[0]*500, ub=[1]*500, precision=1e-3)
ga.run()
print('best_x is ', ga.best_x, 'best_y is', ga.best_y)

render_fn=lambda params: painter.render(params, background='white')

# img = arr2img(render_fn(pso.gbest_x)) PSO
img = arr2img(render_fn(ga.best_x))
# pylint:disable=undefined-variable
fig, ax = plt.subplots()
ax.imshow(img)
plt.show()
