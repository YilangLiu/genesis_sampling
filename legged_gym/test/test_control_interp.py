import torch 
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
import matplotlib.pyplot as plt
import numpy as np 


def control_interpolations(x: torch.Tensor, y: torch.Tensor, interp_array: torch.Tensor):
    # torch_array (num_samples, num_knots, num_actions)
    coeffs = natural_cubic_spline_coeffs(x, y)
    spline = NaturalCubicSpline(coeffs)
    return spline.evaluate(interp_array) 


T = 100
num_knots = 10
ctrl_dt = 0.02
num_samples = 10
num_actions = 12

ctrl_steps = torch.linspace(0, T * ctrl_dt, T, dtype=torch.float32)
knots_steps = torch.linspace(0, T * ctrl_dt, num_knots, dtype=torch.float32)

ctrl_knots = torch.repeat_interleave(torch.sin(knots_steps.reshape((1, -1, 1))), num_samples, dim=0)

ctrls = control_interpolations(knots_steps, ctrl_knots, ctrl_steps)

import pdb; pdb.set_trace()


plt.plot(ctrl_steps, ctrls[0,:,0], 'b', label="before time shift")
plt.plot(knots_steps, ctrl_knots[0,:,0], 'r*')

ctrls_shift = control_interpolations(knots_steps, ctrl_knots, knots_steps + 10 * ctrl_dt)
plt.plot(knots_steps+ 10 * ctrl_dt, ctrls_shift[0,:,0], 'g*')




plt.show()


