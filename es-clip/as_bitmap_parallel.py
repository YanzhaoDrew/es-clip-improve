import random
import tkinter as tk
from PIL import Image, ImageTk, ImageChops
from utils import load_target, save_as_gif, img2tensor
from tqdm import *
import torch
import clip
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device_index = 'cuda:0'
INIT_POPULATION = 1
INIT_BATCH_SIZE = 128
ITERATION = 10000

# Constants
ACTUAL_SHAPES = 80
# ACTUAL_POINTS = 6
IWIDTH = 200
IHEIGHT = 200
NORM_COEF = 1.0
MAX_POINTS = 3
# MAX_SHAPES = 50
MAX_COLOR_SHAPES = 4

# Global variables (these need to be initialized properly)
COST_TEST = 0
COST_BEST = float('inf')
FITNESS_BEST_NORMALIZED = 0
FITNESS_BEST_RECORD = []
CHANGED_SHAPE_INDEX = 0
COUNTER_BENEFIT = 0
COUNTER_TOTAL = 0
LAST_START = 0
ELAPSED_TIME = 0
LAST_COUNTER = 0

# INIT
INIT_TYPE = 'else'
INIT_A = 0
INIT_G = 0
INIT_B = 0
INIT_R = 0


# GUI elements (replace these with actual Tkinter elements)
EL_FITNESS = None
EL_STEP_BENEFIT = None
EL_STEP_TOTAL = None
EL_ELAPSED_TIME = None
EL_MUTSEC = None
target_image = img2tensor(load_target('assets/monalisa.png', (IHEIGHT, IWIDTH)))
images = []
from PIL import Image, ImageDraw

def drawDNA(dna, width, height):
    dna = dna.tolist()
    image = Image.new('RGBA', (width, height), color='white')
    draw = ImageDraw.Draw(image, 'RGBA')

    for i in range(len(dna)):
        drawShape(draw, dna[i][:MAX_POINTS*2], dna[i][MAX_POINTS*2:])

    return image

def drawShape(draw, shape, color):
    # shape shape(MAX_POINTS * 2)
    # color shape(4)
    points = []
    for i in range(MAX_POINTS):
        x = int(shape[i])
        y = int(shape[MAX_POINTS + i])
        points.append((x, y))

    color = tuple(int(c) for c in color)
    draw.polygon(points, fill=color)

def rand_int(max_val):
    return random.randint(0, max_val)

def rand_float(max_val):
    return random.uniform(0, max_val)

def get_timestamp():
    import time
    return int(time.time())

def render_nice_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"

def compute_fitness(dna):
    # 根据DNA绘制图像
    dna_images = torch.zeros(dna.size(0), 4, IHEIGHT, IWIDTH)
    for b in range(dna.size(0)):
        dna_images[b, :, :, :] = img2tensor(drawDNA(dna[b], IHEIGHT, IWIDTH))

    target_image_arr = target_image.unsqueeze(0)#.repeat(dna.size(0), 1, 1, 1)

    # 将绘制的图像与目标图像做差
    l2_loss_per_pixel = torch.pow(dna_images/255.0 - target_image_arr/255.0, 2)
    l2_loss_per_population = torch.mean(l2_loss_per_pixel, dim=(1, 2, 3))

    return l2_loss_per_population.to('cpu').tolist()


def init_dna(dna):
    # dna.shape = (INIT_POPULATION, MAX_SHAPES, MAX_POINTS * 2 + 4)
    for p in range(dna.size(0)):
        for i in range(ACTUAL_SHAPES):
            dna[p, i, :MAX_POINTS] = torch.randint(0, IWIDTH, (MAX_POINTS,), device=device)
            dna[p, i, MAX_POINTS:] = torch.randint(0, IHEIGHT, (MAX_POINTS,), device=device)

    # Initialize color tensor
    if INIT_TYPE == "random":
        color = torch.randint(0, 256, (dna.size(0), ACTUAL_SHAPES, 4), device=device).float()
        # color[:, :, 3] = 0.001
    else:
        color = torch.tensor([INIT_R, INIT_G, INIT_B, INIT_A], dtype=torch.float32, device=device)
        color = color.repeat(dna.size(0), ACTUAL_SHAPES, 1)

    # Concatenate color tensor to dna tensor
    dna = torch.cat((dna, color), dim=2)
    return dna

def pass_gene_mutation(dna_from, dna_to, gene_index):
    # dna_from, dna_to. shape(INIT_POPULATION, MAX_SHAPES, MAX_POINTS * 2 + 4)
    # dna_to[ gene_index, MAX_POINTS * 2:] = dna_from[ gene_index, MAX_POINTS * 2:]
    # dna_to[ gene_index, :MAX_POINTS * 2] = dna_from[ gene_index, :MAX_POINTS * 2]
    dna_to[ gene_index] = dna_from[ gene_index]

def mutate_medium(dna_out):
    # dna_out shape(INIT_POPULATION, MAX_SHAPES, MAX_POINTS * 2 + 4)
    global CHANGED_SHAPE_INDEX
    CHANGED_SHAPE_INDEX = rand_int(ACTUAL_SHAPES - 1)

    roulette = rand_float(2.0)
    batch_size = dna_out.size(0)

    if roulette < 1:
        if roulette < 0.25:
            dna_out[:, CHANGED_SHAPE_INDEX, MAX_POINTS * 2] = torch.randint(0, 256, (batch_size,), device=device)
            
        elif roulette < 0.5:
            dna_out[:, CHANGED_SHAPE_INDEX, MAX_POINTS * 2 + 1] = torch.randint(0, 256, (batch_size,), device=device)
        elif roulette < 0.75:
            dna_out[:, CHANGED_SHAPE_INDEX, MAX_POINTS * 2 + 2] = torch.randint(0, 256, (batch_size,), device=device)
        else:
            dna_out[:, CHANGED_SHAPE_INDEX, MAX_POINTS * 2 + 3] = torch.randint(0, 256, (batch_size,), device=device)
    else:
        CHANGED_POINT_INDEX = rand_int(MAX_POINTS - 1)

        if roulette < 1.5:
            dna_out[:, CHANGED_SHAPE_INDEX, CHANGED_POINT_INDEX] = torch.randint(0, IWIDTH+1, (batch_size,), device=device)
        else:
            dna_out[:, CHANGED_SHAPE_INDEX, MAX_POINTS + CHANGED_POINT_INDEX] = torch.randint(0, IHEIGHT+1, (batch_size,), device=device)

def evolve():
    global COST_TEST, COST_BEST, FITNESS_BEST_NORMALIZED, COUNTER_BENEFIT, COUNTER_TOTAL, LAST_START, LAST_COUNTER, ELAPSED_TIME, prior_dna_size
    train_data = TensorDataset(DNA_TEST)
    dataset = DataLoader(train_data, batch_size=INIT_BATCH_SIZE, shuffle=False, num_workers=0)
    prior_dna_size = 0
    for id, dna in enumerate(dataset):
        mutateDNA(dna[0])
        # drawDNA(DNA_TEST, IHEIGHT, IWIDTH)

        COST_TEST = compute_fitness(dna[0])
        min_COST_TEST_index = COST_TEST.index(min(COST_TEST))

        if COST_TEST[min_COST_TEST_index] < COST_BEST:
            pass_gene_mutation(dna[0][min_COST_TEST_index], DNA_BEST[0], CHANGED_SHAPE_INDEX)
            # indices = torch.where(
                # torch.tensor(COST_TEST) > torch.tensor(COST_BEST[prior_dna_size:prior_dna_size + dna[0].size(0)]))
            # 更新满足条件的 DNA_BEST
            COST_BEST = COST_TEST[min_COST_TEST_index]
            FITNESS_BEST_NORMALIZED = 100 * (1 -COST_BEST)
            COUNTER_BENEFIT += 1

            images.append(drawDNA(DNA_BEST[0], IHEIGHT, IWIDTH))
        else:
            pass_gene_mutation(DNA_BEST[0], dna[0][min_COST_TEST_index], CHANGED_SHAPE_INDEX)

        COUNTER_TOTAL += 1
        DNA_TEST[prior_dna_size : prior_dna_size + dna[0].size(0)] = dna[0]
        # EL_STEP_TOTAL.config(text=str(COUNTER_TOTAL))

        # if COUNTER_BENEFIT % 10 == 0:
        #     print('有', COUNTER_BENEFIT)


        if COUNTER_TOTAL % 1 == 0:
            FITNESS_BEST_RECORD.append(FITNESS_BEST_NORMALIZED)
        prior_dna_size += dna[0].size(0)

# Placeholder for mutateDNA function
def mutateDNA(dna):
    mutate_medium(dna)


if __name__ == "__main__":
    global worker_assets
    device = torch.device(device_index)
    worker_assets = {
        'device': device,
    }
    global DNA_TEST, DNA_BEST
    DNA_TEST = torch.zeros((INIT_POPULATION, ACTUAL_SHAPES, MAX_POINTS * 2), dtype=torch.float32, device=device)
    DNA_BEST = torch.zeros((1, ACTUAL_SHAPES, MAX_POINTS * 2), dtype=torch.float32, device=device)

    DNA_TEST = init_dna(DNA_TEST.clone())
    DNA_BEST = init_dna(DNA_BEST.clone())
    # print(DNA_TEST)
    for i in tqdm(range(ITERATION)):
        evolve()
    save_as_gif(f'as_bitmap_out/test_as_bitmap-ITER{ITERATION}-POP{INIT_POPULATION}-MAXPOINT{MAX_POINTS}-BATCH{INIT_BATCH_SIZE}-ACTUAL_SHAPES{ACTUAL_SHAPES}.gif', images, fps=len(images) / 5)
    Image._show(images[-1])
    images[-1].save(f"as_bitmap_out/test_as_bitmap-ITER{ITERATION}-POP{INIT_POPULATION}-MAXPOINT{MAX_POINTS}-BATCH{INIT_BATCH_SIZE}-ACTUAL_SHAPES{ACTUAL_SHAPES}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(FITNESS_BEST_RECORD)), FITNESS_BEST_RECORD, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness Over Generations')
    plt.grid(True)
    plt.savefig(f'as_bitmap_out/fitness_as_bitmap-ITER{ITERATION}-POP{INIT_POPULATION}-MAXPOINT{MAX_POINTS}-BATCH{INIT_BATCH_SIZE}-ACTUAL_SHAPES{ACTUAL_SHAPES}.png')
    plt.show()