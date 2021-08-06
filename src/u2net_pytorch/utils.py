import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob

from .data_loader import RescaleT
from .data_loader import ToTensorLab
from .data_loader import SalObjDataset

from .model import U2NET  # full size version 173.6 MB
from .model import U2NETP  # small version u2net 4.7 MB


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir, prefix):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert("RGB")
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    aaa = img_name.split(".")
    imidx = ".".join(aaa[0:-1])

    imo.save(os.path.join(d_dir, prefix + imidx + ".png"))


def run_inference(
    data_dirpath, prediction_dirpath=None, model_name="u2net", prefix=None
):

    image_dir = data_dirpath
    prediction_dir = data_dirpath if prediction_dirpath is None else prediction_dirpath
    prefix = prefix if prefix is not None else "saliency-map_" + model_name + "_"
    os.makedirs(prediction_dir, exist_ok=True)

    filepath = os.path.realpath(__file__)
    file_dirpath = os.path.dirname(filepath)
    model_dir = os.path.join(
        file_dirpath, "saved_models", model_name, model_name + ".pth"
    )

    img_name_list = glob.glob(image_dir + os.sep + "*")

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]),
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    # --------- 3. model define ---------
    if model_name == "u2net":
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif model_name == "u2netp":
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    else:
        raise ValueError(
            "model_name parameter must be of u2net or u2netp. Instead, "
            + model_name
            + " was passed."
        )

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location="cpu"))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test["image"]
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        save_output(img_name_list[i_test], pred, prediction_dir, prefix)

        del d1, d2, d3, d4, d5, d6, d7
