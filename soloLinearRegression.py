from linearRegression.models import get_model
from linearRegression.utils import fit_model, get_loss, get_params
from linearRegression.plots import plot_scatter_line
from linearRegression.utils import get_dataset, get_dataset_shape

columnIdx = 3
mode = "sklearn"

data_X, data_Y = get_dataset(columnIdx)
inDim, outDim = get_dataset_shape(columnIdx)

model = get_model(mode, inputDim = inDim, outputDim = outDim)
if(mode == "keras"):
    history = fit_model(mode, model, X = data_X, Y = data_Y)
else:
    model = fit_model(mode, model, X = data_X, Y = data_Y)

print(f"{mode} Loss: {get_loss(mode, model, X = data_X, Y = data_Y, pred_Y = model.predict(data_X), target_Y = data_Y)}")
weight, bias = get_params(mode, model)
print(f"{mode} Weight: {weight}")
print(f"{mode} Bias: {bias}")
plot_scatter_line(data_X, data_Y, weight, bias)