import os
from basicsr.data.paired_image_dataset import PairedImageDataset
from basicsr.utils import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageDatasetWithAttention(PairedImageDataset):
    """Paired image dataset with attention maps."""

    def __init__(self, opt):
        super().__init__(opt)
        self.dataroot_attention = opt['dataroot_attention']
        self.attention_tmpl = opt.get('attention_tmpl', '{}_attention')

    def __getitem__(self, index):
        # Get original data
        data = super().__getitem__(index)

        # Load attention map
        attention_path = self.get_attention_path(data['lq_path'])
        attention = self.read_attention(attention_path)

        # Add attention to data
        data['attention'] = attention

        return data

    def get_attention_path(self, lq_path):
        """Get attention map path from LQ image path."""
        basename = os.path.splitext(os.path.basename(lq_path))[0]
        attention_filename = self.attention_tmpl.format(basename) + '.png'
        return os.path.join(self.dataroot_attention, attention_filename)

    def read_attention(self, attention_path):
        """Read and process attention map."""
        # Implement your attention map reading logic
        # This could be a numpy array, tensor, etc.
        # Example: load as grayscale image and normalize
        import cv2
        attention = cv2.imread(attention_path, cv2.IMREAD_GRAYSCALE)
        attention = attention.astype(np.float32) / 255.0
        attention = img2tensor(attention, bgr2rgb=False, float32=True)
        return attention