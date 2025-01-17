import random
import tkinter as tk
from PIL import Image, ImageTk, ImageChops
from utils import load_target, rgba2rgb, save_as_gif, img2tensor
from tqdm import *
import torch
import clip
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device_index = 'cuda:0'
INIT_POPULATION = 50
INIT_BATCH_SIZE = 128
ITERATION = 1000

# Constants
ACTUAL_SHAPES = 50
ACTUAL_POINTS = 6
IWIDTH = 200
IHEIGHT = 200
NORM_COEF = 1.0
MAX_POINTS = 6
MAX_SHAPES = 50
MAX_COLOR_SHAPES = 4

# Global variables (these need to be initialized properly)
FITNESS_TEST = 0
FITNESS_BEST = 0.
FITNESS_BEST_NORMALIZED = 0
FITNESS_BEST_RECORD = []
CHANGED_SHAPE_INDEX = 0
COUNTER_BENEFIT = 0
COUNTER_TOTAL = 0
LAST_START = 0
ELAPSED_TIME = 0
LAST_COUNTER = 0

# INIT
INIT_TYPE = 'random'
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
# target_image = load_target('assets/monalisa.png', (IHEIGHT, IWIDTH))
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
    device = worker_assets['device']
    model = worker_assets['model']
    text_features = worker_assets['text_features']
    NUM_AUGS = 4

    # 根据DNA绘制图像
    dna_images = torch.zeros(dna.size(0), 3, IHEIGHT, IWIDTH)
    for b in range(dna.size(0)):
        dna_images[b, :, :, :] = img2tensor(rgba2rgb( drawDNA(dna[b], IHEIGHT, IWIDTH)))

    n_solution = dna.size(0)
    with torch.no_grad():
        t = torch.tensor(dna_images.clone()).to(device)
        t = t.type(torch.float32) / 255.
        t = t.repeat_interleave(NUM_AUGS, dim=0)  # 增强后为 (128 * 4, 3, 200, 200)
        new_augment_trans = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        t = new_augment_trans(t)
        im_batch = t
        image_features = model.encode_image(im_batch)  # shape(512,512)
        similiarities = torch.cosine_similarity(image_features, text_features, axis=-1)  # shape(512,)
        similiarities = torch.reshape(similiarities, (n_solution, NUM_AUGS)).mean(axis=-1)  # shape(128,)

    similiarities = similiarities.to('cpu').tolist()  # list len=128
    # print(similiarities)
    return similiarities

    # 将绘制的图像与目标图像做差
    # diff = ImageChops.difference(target_image, dna_image)

    # 将差异图像的像素值元组转换为整数,并计算总和
    # fitness = sum(sum(pixel) for pixel in diff.getdata())
    #
    # return fitness


def init_dna(dna):
    # dna.shape = (INIT_POPULATION, MAX_SHAPES, MAX_POINTS * 2 + 4)
    for p in range(dna.size(0)):
        for i in range(MAX_SHAPES):
            dna[p, i, :MAX_POINTS] = torch.randint(0, IWIDTH, (MAX_POINTS,), device=device)
            dna[p, i, MAX_POINTS:] = torch.randint(0, IHEIGHT, (MAX_POINTS,), device=device)

    # Initialize color tensor
    if INIT_TYPE == "random":
        color = torch.randint(0, 256, (dna.size(0), MAX_SHAPES, 4), device=device).float()
        # color[:, :, 3] = 0.001
    else:
        color = torch.tensor([INIT_R, INIT_G, INIT_B, INIT_A], dtype=torch.float32, device=device)
        color = color.repeat(dna.size(0), MAX_SHAPES, 1)

    # Concatenate color tensor to dna tensor
    dna = torch.cat((dna, color), dim=2)
    return dna

def pass_gene_mutation(dna_from, dna_to, gene_index):
    # dna_from, dna_to. shape(INIT_POPULATION, MAX_SHAPES, MAX_POINTS * 2 + 4)
    # dna_to[:, gene_index, MAX_POINTS * 2:] = dna_from[:, gene_index, MAX_POINTS * 2:]
    # dna_to[:, gene_index, :MAX_POINTS * 2] = dna_from[:, gene_index, :MAX_POINTS * 2]
    dna_to[ gene_index] = dna_from[ gene_index]


def mutate_medium(dna_out):
    # dna_out shape(INIT_POPULATION, MAX_SHAPES, MAX_POINTS * 2 + 4)
    global CHANGED_SHAPE_INDEX
    CHANGED_SHAPE_INDEX = rand_int(MAX_SHAPES - 1)

    roulette = rand_float(2.0)

    if roulette < 1:
        if roulette < 0.25:
            dna_out[:, CHANGED_SHAPE_INDEX, MAX_POINTS * 2] = rand_int(255)
        elif roulette < 0.5:
            dna_out[:, CHANGED_SHAPE_INDEX, MAX_POINTS * 2 + 1] = rand_int(255)
        elif roulette < 0.75:
            dna_out[:, CHANGED_SHAPE_INDEX, MAX_POINTS * 2 + 2] = rand_int(255)
        else:
            dna_out[:, CHANGED_SHAPE_INDEX, MAX_POINTS * 2 + 3] = rand_int(255)
    else:
        CHANGED_POINT_INDEX = rand_int(MAX_POINTS - 1)

        if roulette < 1.5:
            dna_out[:, CHANGED_SHAPE_INDEX, CHANGED_POINT_INDEX] = rand_int(IWIDTH)
        else:
            dna_out[:, CHANGED_SHAPE_INDEX, MAX_POINTS + CHANGED_POINT_INDEX] = rand_int(IHEIGHT)

def evolve():
    global prior_dna_size, FITNESS_TEST, FITNESS_BEST, FITNESS_BEST_NORMALIZED, COUNTER_BENEFIT, COUNTER_TOTAL, LAST_START, LAST_COUNTER, ELAPSED_TIME
    prior_dna_size = 0
    train_data = TensorDataset(DNA_TEST)
    dataset = DataLoader(train_data, batch_size=INIT_BATCH_SIZE, shuffle=False, num_workers=0)
    for id, dna in enumerate(dataset):
        mutateDNA(dna[0])
        # drawDNA(DNA_TEST, IHEIGHT, IWIDTH)

        FITNESS_TEST = compute_fitness(dna[0])
        max_FITNESS_TEST_index = FITNESS_TEST.index(max(FITNESS_TEST))

        if FITNESS_TEST[max_FITNESS_TEST_index] > FITNESS_BEST:
            pass_gene_mutation(dna[0][max_FITNESS_TEST_index], DNA_BEST[0], CHANGED_SHAPE_INDEX)
            # indices = torch.where(torch.tensor(FITNESS_TEST) > torch.tensor(FITNESS_BEST))
            # 更新满足条件的 DNA_BEST
            # DNA_BEST[prior_dna_size + indices[0]] = dna[0][indices[0]]
            FITNESS_BEST = FITNESS_TEST[max_FITNESS_TEST_index]
            FITNESS_BEST_NORMALIZED = 100 * FITNESS_BEST
            # print(FITNESS_BEST)
            # EL_FITNESS.config(text=f"{FITNESS_BEST_NORMALIZED:.2f}%")
            COUNTER_BENEFIT += 1
            # EL_STEP_BENEFIT.config(text=str(COUNTER_BENEFIT))
            images.append(drawDNA(DNA_BEST[0], IHEIGHT, IWIDTH))
        else:
            pass_gene_mutation(DNA_BEST[0], dna[0][max_FITNESS_TEST_index], CHANGED_SHAPE_INDEX)

        COUNTER_TOTAL += 1
        DNA_TEST[prior_dna_size:prior_dna_size + dna[0].size(0)] = dna[0]
        # EL_STEP_TOTAL.config(text=str(COUNTER_TOTAL))

        # if COUNTER_TOTAL % 10 == 0:
        #     passed = get_timestamp() - LAST_START
        #     # print('paseed:', passed)
        #     # EL_ELAPSED_TIME.config(text=render_nice_time(ELAPSED_TIME + passed))

        if COUNTER_TOTAL % 1 == 0:
            FITNESS_BEST_RECORD.append(FITNESS_BEST_NORMALIZED)
        prior_dna_size += dna[0].size(0)

# Placeholder for mutateDNA function
def mutateDNA(dna):
    mutate_medium(dna)


if __name__ == "__main__":
    global worker_assets
    device = torch.device(device_index)
    text = input("输入文字描述:")
    model, preprocess = clip.load('ViT-B/32', device, jit=True)  # change jit=True
    text_input = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    worker_assets = {
        'device': device,
        'model': model,
        'preprocess': preprocess,
        'text_features': text_features,
    }
    global DNA_TEST, DNA_BEST
    DNA_TEST = torch.zeros((INIT_POPULATION, ACTUAL_SHAPES, MAX_POINTS * 2), dtype=torch.float32, device=device)
    DNA_BEST = torch.zeros((1, ACTUAL_SHAPES, MAX_POINTS * 2), dtype=torch.float32, device=device)

    DNA_TEST = init_dna(DNA_TEST.clone())
    DNA_BEST = init_dna(DNA_BEST.clone())
    # print(DNA_TEST)
    for i in tqdm(range(ITERATION)):
        evolve()
    save_as_gif(f'test_as_clip-text{text}-ITER{ITERATION}-POP{INIT_POPULATION}-MAXPOINT{MAX_POINTS}-BATCH{INIT_BATCH_SIZE}.gif', images, fps=8)
    Image._show(images[-1])
    images[-1].save(f"test_as_clip-text{text}-ITER{ITERATION}-POP{INIT_POPULATION}-MAXPOINT{MAX_POINTS}-BATCH{INIT_BATCH_SIZE}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(FITNESS_BEST_RECORD)), FITNESS_BEST_RECORD, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness Over Generations')
    plt.grid(True)
    plt.show()