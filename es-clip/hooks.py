#!/usr/bin/env python3

from datetime import datetime
from utils import (isnotebook, tensor2img, save_as_gif, save_as_frames)

class Hook(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        pass


class PrintStepHook(Hook):
    def __init__(self):
        super().__init__()

    def __call__(self, i, **kwargs):
        print(i, end=' ... ')


class PrintCostHook(Hook):
    def __init__(self):
        super().__init__()

    def __call__(self, i, solver, fitnesses_fn, best_params_fn, **kwargs):
        best_params = best_params_fn(solver)
        # if self.fitnesses_fn_is_wrapper:
        #     cost = fitnesses_fn(fitness_fn, [best_params])
        # else:
            
        cost = fitnesses_fn([best_params])
        print()
        print(f'[{datetime.now()}]   Iteration: {i}   cost: {cost}')


class SaveCostHook(Hook):
    def __init__(self, save_fp):
        super().__init__()
        self.save_fp = save_fp
        self.record = []  # list of (i, cost)

    def __call__(self, i, solver, fitnesses_fn, best_params_fn, **kwargs):
        best_params = best_params_fn(solver)
        # if self.fitnesses_fn_is_wrapper:
        #     cost = fitnesses_fn(fitness_fn, [best_params])
        # else:
        
        cost = fitnesses_fn([best_params])
        self.record.append(f'[{datetime.now()}]   Iteration: {i}   cost: {cost}')
        with open(self.save_fp, 'w') as fout:
            list(map(lambda r: print(r, file=fout), self.record))


class StoreImageHook(Hook):
    def __init__(self, render_fn, save_fp, fps=12, save_interval=0):
        super().__init__()
        self.render_fn = render_fn
        self.save_fp = save_fp
        self.fps = fps
        self.save_interval = save_interval

        self.imgs = []

    def __call__(self, i, solver, best_params_fn, **kwargs):
        best_params = best_params_fn(solver)
        img = tensor2img(self.render_fn(best_params))
        self.imgs.append(img)
        if i % self.save_interval == 0:
            self.save()

    # def __del__(self):
    #     self.save()

    def save(self):
        save_as_gif(f'{self.save_fp}.gif', self.imgs, fps=self.fps)
        save_as_frames(f'{self.save_fp}.frames', self.imgs, overwrite=False)


class ShowImageHook(Hook):
    def __init__(self, render_fn):
        super().__init__()
        self.render_fn = render_fn

    def __call__(self, solver, best_params_fn, **kwargs):
        if isnotebook():
            best_params = best_params_fn(solver)
            img = tensor2img(self.render_fn(best_params))
            # pylint:disable=undefined-variable
            display(img)  # type: ignore
