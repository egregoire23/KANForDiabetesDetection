import torch
import torch.nn as nn

class LinearSpline(nn.Module):
    def __init__(self, knots, x_min, x_max):
        super().__init__()
        self.knots = knots
        self.x_min =x_min
        self.x_max= x_max
        self.register_buffer("knot_x", torch.linspace(self.x_min, self.x_max, knots))

        self.knot_y = nn.Parameter(torch.rand(self.knots))

    def forward(self, x):
        idx = torch.bucketize(x, self.knot_x)-1
        left_bound = self.knot_x[idx]
        right_bound = self.knot_x[idx+1]
        distance = (x-left_bound)/(right_bound-left_bound)
        left_y=self.knot_y[idx]
        right_y = self.knot_y[idx+1]
        interpolated_y = torch.lerp(left_y, right_y, distance)
        return interpolated_y


    
class CubicSpline(nn.Module):
    def __init__(self, knots, x_min, x_max):
        super().__init__()
        self.knots = knots
        self.x_min =x_min
        self.x_max= x_max
        self.register_buffer("knot_x", torch.linspace(self.x_min, self.x_max, knots))

        self.knot_y = nn.Parameter(torch.rand(self.knots))

    def forward(self, x):
        knot_x = self.knot_x.to(x.device)
        idx = torch.bucketize(x, self.knot_x)-1
        idx = idx.clamp(1,self.knots-3)
        x_one= self.knot_x[idx-1]
        x_second= self.knot_x[idx]
        x_third= self.knot_x[idx+1]
        first= self.knot_y[idx-1]
        second= self.knot_y[idx]
        third= self.knot_y[idx+1]
        forth= self.knot_y[idx+2]
        cubic_coefficents = (x-x_second)/(x_third -x_second)
        #cubic interpolation used here is the catmull formula

        interpolated_y = .5*((2*second)+(-first+third)*cubic_coefficents +(2*first-5*second+4*third - forth)*cubic_coefficents**2 + 
                            (-first + 3*second-3*third+forth)*cubic_coefficents**3)
        return interpolated_y



class KANLayer(nn.Module):
    def __init__(self,in_features, out_features, feature_bounds=None):
        super().__init__()
        self.in_features=in_features
        self.out_features = out_features
        self.splineholder =nn.ModuleList()
        if feature_bounds !=None:
            for i in range(in_features):
                x_min, x_max = feature_bounds[i]
                self.splineholder.append(CubicSpline(10, x_min, x_max))
        else:
            for i in range(in_features):
                self.splineholder.append(CubicSpline(50,-10,10))
        self.combo_layer = nn.Linear(self.in_features, self.out_features)

    def forward(self,x):
        spline_outs=[]
        for i in range(self.in_features):
            transformation = self.splineholder[i](x[:,i])
            spline_outs.append(transformation)

        transformed_x = torch.stack(spline_outs, dim=1)
        combination = self.combo_layer(transformed_x)
        return combination
    



