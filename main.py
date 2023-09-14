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
            print("file_name = ", filename)
            args.ref_image = "./src_image/{}/{}".format(style, filename)
            image_editor = ImageEditor(args)
            image_editor.edit_image_by_hybrid()
            #image_editor.edit_image_by_image_prompt()
            torch.cuda.empty_cache()
            # image_editor.edit_image_by_image()
            torch.cuda.empty_cache()
            # image_editor.edit_image_by_prompt()
            image_editor.save_image()
            matrix["clip_prompt"].append(image_editor.matrix["clip_prompt"][0])
            matrix["clip_image"].append(image_editor.matrix["clip_image"][0])
            matrix["clip_gram"].append(image_editor.matrix["clip_gram"][0])
            matrix["gram_prompt"].append(image_editor.matrix["gram_prompt"][0])
            matrix["gram_image"].append(image_editor.matrix["gram_image"][0])
            matrix["gram_gram"].append(image_editor.matrix["gram_gram"][0])
            matrix["hybrid_prompt"].append(
                image_editor.matrix["hybrid_prompt"][0])
            matrix["hybrid_gram"].append(image_editor.matrix["hybrid_gram"][0])
            matrix["hybrid_image"].append(
                image_editor.matrix["hybrid_image"][0])
        index += 1
    file_path = 'matrix_{}_{}_{}.pickle'.format(style, args.begin, args.end)
    with open(file_path, 'wb') as file:
        pickle.dump(matrix, file)
