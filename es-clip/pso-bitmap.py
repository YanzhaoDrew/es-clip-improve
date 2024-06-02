from painter import TrianglesPainter
from es_bitmap import load_target, EasyDict, PrintCostHook, PrintStepHook, SaveCostHook, StoreImageHook, ShowImageHook
import numpy as np
import os
from utils import arr2img,save_as_gif, save_as_frames
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
height, width = 200, 200
global painter, target_arr, loss_type
painter = TrianglesPainter(h=200, w=200, n_triangle=50, alpha_scale=0.1, coordinate_scale=1.0)
target_arr = load_target('assets/monalisa.png', (height, width))
loss_type = 'l2'
imgs = []



def parse_args(cmd_args=None):
    args = EasyDict()
    out_dir = 'es_bitmap_out'
    fps = 12
    report_interval = 50
    step_report_interval = 50
    save_as_gif_interval = 10
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
pso = PSO(func=fitness_fn, n_dim=500, pop=40, max_iter=100, lb=[0]*500, ub=[1]*500, w=0.8, c1=0.5, c2=0.5)
pso.record_mode = True
pso.run()
history_x = pso.record_value['X']
render_fn = lambda params: painter.render(params, background='white')
save_fp = os.path.join(args.working_dir, 'animate-background=white')
for i, x in enumerate(history_x):
    img = arr2img(render_fn(x))
    imgs.append(img)
save_as_gif(f'{save_fp}.gif', imgs, fps=args.fps)
save_as_frames(f'{save_fp}.frames', imgs, overwrite=False)
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
img = arr2img(render_fn(pso.gbest_x))
# pylint:disable=undefined-variable
fig, ax = plt.subplots()
ax.imshow(img)
plt.show()
