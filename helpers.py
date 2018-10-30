import os.path
import cv2

class PipelineDebug:
    """ Helps output intermediate images while executing the pipeline """
    def __init__(self, img_path, pipeline_folder):
        self.img_path = os.path.basename(img_path)
        self.img_name, self.img_ext = os.path.splitext(self.img_path)
        self.pipeline_folder = pipeline_folder
        self.step = 0
        self.enable = True

    def inc_step(self):
        self.step += 1
        return self.step - 1
    
    def s(self, img, desc, *args, **kwargs):
        if self.enable == False: return
        file_path = os.path.join(self.pipeline_folder, "{name}_Step{stp:02d}_{desc}{ext}".format(
            name=self.img_name,
            stp=self.inc_step(),
            desc=desc, 
            ext=self.img_ext))
        cv2.imwrite(file_path, img, *args, **kwargs)