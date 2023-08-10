from optimization.image_editor_zecon import ImageEditor
from optimization.arguments import get_arguments
import torch
import pickle
if __name__ == "__main__":

    args = get_arguments()
    image_editor = ImageEditor(args)
    image_editor.edit_image_by_image_prompt()
    torch.cuda.empty_cache()
    image_editor.edit_image_by_image()
    torch.cuda.empty_cache()
    image_editor.edit_image_by_prompt()
    image_editor.save_image()
    file_path = 'matrix.pickle'
    with open(file_path, 'wb') as file:
        pickle.dump(image_editor.matrix, file)
