import random
import tkinter as tk
from PIL import Image, ImageTk, ImageChops
from utils import load_target, save_as_gif
from tqdm import *
import matplotlib.pyplot as plt
from torch import Tensor

# Constants
ACTUAL_SHAPES = 50
ACTUAL_POINTS = 6
IWIDTH = 200
IHEIGHT = 200
NORM_COEF = 1.0
MAX_POINTS = 6
MAX_SHAPES = 50

# Global variables (these need to be initialized properly)
DNA_TEST = []
DNA_BEST = []
COST_TEST = 0
COST_BEST = float('inf')
FITNESS_BEST_RECORD = []
FITNESS_BEST_NORMALIZED = 0
CHANGED_SHAPE_INDEX = 0
COUNTER_BENEFIT = 0
COUNTER_TOTAL = 0
LAST_START = 0
ELAPSED_TIME = 0
LAST_COUNTER = 0

# INIT
INIT_TYPE = 'color'
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
target_image = load_target('assets/monalisa.png', (IHEIGHT, IWIDTH))
images = []
from PIL import Image, ImageDraw


def drawDNA(dna, width, height):
    image = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(image)

    for i in range(len(dna)):
        drawShape(draw, dna[i]['shape'], dna[i]['color'])

    return image


def drawShape(draw, shape, color):
    points = [(point['x'], point['y']) for point in shape]
    draw.polygon(points, fill=tuple(color.values()))


def rand_int(max_val):
    return random.randint(0, max_val)


def rand_float(max_val):
    return random.uniform(0, max_val)


def compute_fitness(dna):
    # 根据DNA绘制图像
    dna_image = drawDNA(dna, IHEIGHT, IWIDTH)

    # 将绘制的图像与目标图像做差
    diff = ImageChops.difference(target_image, dna_image)

    # 将差异图像的像素值元组转换为整数,并计算总和
    fitness = sum(sum(pixel) for pixel in diff.getdata())
    # diff_pixels = Tensor(diff.getdata())
    # fitness = diff_pixels.sum()
    # diff.getdata()

    return fitness


def init_dna(dna):
    for i in range(MAX_SHAPES):
        points = []
        for j in range(MAX_POINTS):
            points.append({'x': rand_int(IWIDTH), 'y': rand_int(IHEIGHT)})

        if INIT_TYPE == "random":
            color = {'r': rand_int(255), 'g': rand_int(255), 'b': rand_int(255), 'a': rand_int(255)}
        else:
            color = {'r': INIT_R, 'g': INIT_G, 'b': INIT_B, 'a': INIT_A}

        shape = {
            'color': color,
            'shape': points
        }
        dna[i] = shape


def pass_gene_mutation(dna_from, dna_to, gene_index):
    dna_to[gene_index]['color']['r'] = dna_from[gene_index]['color']['r']
    dna_to[gene_index]['color']['g'] = dna_from[gene_index]['color']['g']
    dna_to[gene_index]['color']['b'] = dna_from[gene_index]['color']['b']
    dna_to[gene_index]['color']['a'] = dna_from[gene_index]['color']['a']

    for i in range(MAX_POINTS):
        dna_to[gene_index]['shape'][i]['x'] = dna_from[gene_index]['shape'][i]['x']
        dna_to[gene_index]['shape'][i]['y'] = dna_from[gene_index]['shape'][i]['y']


def mutate_medium(dna_out):
    global CHANGED_SHAPE_INDEX # 需要变异的多边形的索引
    CHANGED_SHAPE_INDEX = rand_int(ACTUAL_SHAPES - 1)

    roulette = rand_float(2.0)

    if roulette < 1:
        if roulette < 0.25:
            dna_out[CHANGED_SHAPE_INDEX]['color']['r'] = rand_int(255)
        elif roulette < 0.5:
            dna_out[CHANGED_SHAPE_INDEX]['color']['g'] = rand_int(255)
        elif roulette < 0.75:
            dna_out[CHANGED_SHAPE_INDEX]['color']['b'] = rand_int(255)
        else:
            dna_out[CHANGED_SHAPE_INDEX]['color']['a'] = rand_int(255)
    else:
        CHANGED_POINT_INDEX = rand_int(ACTUAL_POINTS - 1)

        if roulette < 1.5:
            dna_out[CHANGED_SHAPE_INDEX]['shape'][CHANGED_POINT_INDEX]['x'] = rand_int(IWIDTH)
        else:
            dna_out[CHANGED_SHAPE_INDEX]['shape'][CHANGED_POINT_INDEX]['y'] = rand_int(IHEIGHT)


def evolve():
    global COST_TEST, COST_BEST, FITNESS_BEST_NORMALIZED, COUNTER_BENEFIT, COUNTER_TOTAL, LAST_START, LAST_COUNTER, ELAPSED_TIME

    mutateDNA(DNA_TEST) # 只变异一个基因
    # drawDNA(DNA_TEST, IHEIGHT, IWIDTH)

    COST_TEST = compute_fitness(DNA_TEST)

    if COST_TEST < COST_BEST:
        pass_gene_mutation(DNA_TEST, DNA_BEST, CHANGED_SHAPE_INDEX)
        COST_BEST = COST_TEST
        FITNESS_BEST_NORMALIZED = 100 * (1 - COST_BEST / 13300000)
        # print(FITNESS_BEST)
        COUNTER_BENEFIT += 1
        images.append(drawDNA(DNA_BEST, IHEIGHT, IWIDTH))
    else:
        pass_gene_mutation(DNA_BEST, DNA_TEST, CHANGED_SHAPE_INDEX)

    COUNTER_TOTAL += 1

    # if COUNTER_TOTAL % 10 == 0:
    #     passed = get_timestamp() - LAST_START
        # print('paseed:', passed)

    if COUNTER_TOTAL % 50 == 0:
        FITNESS_BEST_RECORD.append(FITNESS_BEST_NORMALIZED)
        # mutsec = (COUNTER_TOTAL - LAST_COUNTER) / (get_timestamp() - LAST_START)
        # print('mutsec:', mutsec)


def mutateDNA(dna):
    mutate_medium(dna)


if __name__ == "__main__":
    DNA_TEST = [{'color': {'r': 0, 'g': 0, 'b': 0, 'a': 1.0}, 'shape': [{'x': 0, 'y': 0} for _ in range(MAX_POINTS)]}
                for _ in range(ACTUAL_SHAPES)]
    DNA_BEST = [{'color': {'r': 0, 'g': 0, 'b': 0, 'a': 1.0}, 'shape': [{'x': 0, 'y': 0} for _ in range(MAX_POINTS)]}
                for _ in range(ACTUAL_SHAPES)]
    init_dna(DNA_TEST)
    init_dna(DNA_BEST)
    for i in tqdm(range(10000)):
        evolve()
    save_as_gif('test_as.gif', images, fps=50)
    Image._show(images[-1])

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(FITNESS_BEST_RECORD)), FITNESS_BEST_RECORD, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness Over Generations')
    plt.grid(True)
    plt.show()