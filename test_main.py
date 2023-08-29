from optimization.image_editor_zecon import ImageEditor
from optimization.arguments import get_arguments
import torch
import pickle
import os

if __name__ == "__main__":
    matrix = {}
    matrix["clip_prompt"] = []
    matrix["clip_image"] = []
    matrix["clip_gram"] = []
    matrix["gram_prompt"] = []
    matrix["gram_image"] = []
    matrix["gram_gram"] = []
    matrix["hybrid_prompt"] = []
    matrix["hybrid_gram"] = []
    matrix["hybrid_image"] = []
    args = get_arguments()
    style = args.ref_image
    folder_path = './src_image/{}'.format(style)
    index = 0
    for filename in os.listdir(folder_path):
        if index >= int(args.begin) and index < int(args.end):
            #print("file_name = ", filename)
            args.ref_image = "./src_image/{}/{}".format(style, filename)
            image_editor = ImageEditor(args)
            image_editor.test
        index += 1
