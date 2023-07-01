from optimization.image_editor_zecon import ImageEditor
from optimization.arguments import get_arguments
import torch
if __name__ == "__main__":

    args = get_arguments()
    image_editor = ImageEditor(args)
    image_editor.edit_image_by_prompt()
    torch.cuda.empty_cache()
    image_editor.edit_image_by_image()
    torch.cuda.empty_cache()
    # image_editor.edit_image_by_image_prompt()
    image_editor.save_image()
