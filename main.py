from optimization.image_editor_zecon import ImageEditor
from optimization.arguments import get_arguments
import torch
import pickle
import os

if __name__ == "__main__":
    args = get_arguments()
    style = args.ref_image
    index = 0
    for filename in os.listdir(style):
        if index >= int(args.begin) and index < int(args.end):
            args.ref_image = style + filename
            image_editor = ImageEditor(args)
            image_editor.edit_image_by_hybrid()
            image_editor.save_image()
            torch.cuda.empty_cache()
        index += 1
