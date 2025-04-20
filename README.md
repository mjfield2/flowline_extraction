# flowline_extraction

Extract points along a flow line in a velocity field for analysis and visualization purposes. This is demonstrated using glacier velocity near Thwaites Glacier in Antarctica.

## Usage

* Use the *flowline* function
* You must choose a starting point with coordinates *x0*, *y0*
* choose a direction - forwards or backwards - to advect the points
* choose a mode - time or distance - to control how the points are spaced. In the case of time, the stride is how much time to multiply the velocity components by. For example, the glacier velocity is given in *meters per year* so a stride of 3 would move the point by 3 years in terms of velocity. In the case of distance, the stride is how equally spaced you want the points to be and the function solves for how much to much along each velocity component to get that distance in the hypotenuse. For example, if the stride if 2000 meters, the points will be approximately spaced by 2000 meters.
* Using a combination of forward and backwards you can choose a point of interest and ensure that it is in the flowline, i.e., concatenate to forwards and backwards flowlines.

<img src="./figures/multiple_flowlines.png" width="600"/>

<img src="./figures/flowline_sampling" width="600"/>

<img src="./figures/cross_section.png" width="800"/>
