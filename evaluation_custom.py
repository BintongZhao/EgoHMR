# -*- coding: utf-8 -*-
"""
Custom Evaluation Script for EgoHMR
Supports custom datasets and single-branch feature extraction
"""

import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import time
import json
import pickle
import numpy as np
from torch.utils.data import Dataset
import scipy.io
from PIL import Image
import cv2
import torchvision
import logging
from tqdm import tqdm
from typing import Optional, Dict, List, Union

# Import existing modules
try:
    from configs import get_config, prohmr_config
except ImportError:
    print("Warning: Could not import configs, using fallback")
    def get_config(cfg_path): return None
    def prohmr_config(): return None

try:
    from models import SMPLHead
    from models.backbones import create_backbone
    from models.discriminator import Discriminator
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    SMPLHead = None
    create_backbone = None
    Discriminator = None

try:
    from DiffusionCondition.smpl_diffusion_egohmr import SMPLDiffusion
except ImportError as e:
    print(f"Warning: Could not import SMPLDiffusion: {e}")
    SMPLDiffusion = None

try:
    from tools import AverageMeter, ConsoleLogger
except ImportError as e:
    print(f"Warning: Could not import tools: {e}")
    AverageMeter = None
    ConsoleLogger = None

# Import geometry functions directly
try:
    from utils.geometry import aa_to_rotmat, perspective_projection, rot6d_to_rotmat
except ImportError as e:
    print(f"Warning: Could not import geometry functions: {e}")
    aa_to_rotmat = None
    perspective_projection = None
    rot6d_to_rotmat = None

# Import pose utils directly  
try:
    from utils.pose_utils import Evaluator
except ImportError as e:
    print(f"Warning: Could not import Evaluator: {e}")
    Evaluator = None

# Import evaluation functions
try:
    from train.evaluate_joints import eva_joints, p_mpjpe
except ImportError as e:
    print(f"Warning: Could not import evaluation functions: {e}")
    eva_joints = None
    p_mpjpe = None

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CustomDataset(Dataset):
    """
    Custom dataset class that can load data with flexible image paths and annotation formats
    Supports JSON, MAT, and pickle annotation formats
    """
    
    def __init__(self, 
                 image_dir: str,
                 annotations_path: str = None,
                 image_size: int = 384,
                 stage: str = 'Test',
                 annotation_format: str = 'auto'):
        """
        Initialize custom dataset
        
        Args:
            image_dir: Directory containing images
            annotations_path: Path to annotation file (JSON/MAT/pickle)
            image_size: Target image size for preprocessing
            stage: Dataset stage ('Train', 'Val', 'Test')
            annotation_format: Format of annotations ('json', 'mat', 'pickle', 'auto')
        """
        self.image_dir = image_dir
        self.annotations_path = annotations_path
        self.image_size = image_size
        self.stage = stage
        self.annotation_format = annotation_format
        
        # Get list of images
        self.image_list = []
        self._load_image_list()
        
        # Load annotations if provided
        self.annotations = {}
        self.has_annotations = False
        if annotations_path and os.path.exists(annotations_path):
            self._load_annotations()
            self.has_annotations = True
            
        self.length = len(self.image_list)
        
        # Default values for SMPL parameters availability
        self.has_body_pose = np.ones(self.length, dtype=np.float32)
        self.has_betas = np.ones(self.length, dtype=np.float32)
        
    def _load_image_list(self):
        """Load list of images from directory"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        if os.path.isdir(self.image_dir):
            for filename in sorted(os.listdir(self.image_dir)):
                if any(filename.lower().endswith(ext) for ext in valid_extensions):
                    self.image_list.append(os.path.join(self.image_dir, filename))
        else:
            raise ValueError(f"Image directory not found: {self.image_dir}")
            
    def _load_annotations(self):
        """Load annotations from file"""
        try:
            # Auto-detect format if needed
            if self.annotation_format == 'auto':
                if self.annotations_path.endswith('.json'):
                    self.annotation_format = 'json'
                elif self.annotations_path.endswith('.mat'):
                    self.annotation_format = 'mat'
                elif self.annotations_path.endswith(('.pkl', '.pickle')):
                    self.annotation_format = 'pickle'
                else:
                    raise ValueError(f"Cannot auto-detect annotation format for {self.annotations_path}")
            
            # Load based on format
            if self.annotation_format == 'json':
                with open(self.annotations_path, 'r') as f:
                    self.annotations = json.load(f)
            elif self.annotation_format == 'mat':
                self.annotations = scipy.io.loadmat(self.annotations_path)
            elif self.annotation_format == 'pickle':
                with open(self.annotations_path, 'rb') as f:
                    self.annotations = pickle.load(f)
            else:
                raise ValueError(f"Unsupported annotation format: {self.annotation_format}")
                
        except Exception as e:
            print(f"Warning: Could not load annotations from {self.annotations_path}: {e}")
            self.annotations = {}
    
    def resize_image(self, image, resize_height=None, resize_width=None):
        """Resize image to target dimensions"""
        image_shape = np.shape(image)
        height = image_shape[0]
        width = image_shape[1]
        
        if (resize_height is None) and (resize_width is None):
            return image
        if resize_height is None:
            resize_height = int(height * resize_width / width)
        elif resize_width is None:
            resize_width = int(width * resize_height / height)
            
        image = cv2.resize(image, dsize=(resize_width, resize_height))
        return image
    
    def __getitem__(self, index):
        """Get item from dataset"""
        # Load image
        image_path = self.image_list[index]
        image = Image.open(image_path)
        image = np.array(image)
        
        # Resize image
        image = self.resize_image(image, resize_width=self.image_size, resize_height=self.image_size)
        
        # Apply color augmentation if training
        if self.stage == 'Train':
            image = Image.fromarray(image)
            color_aug = torchvision.transforms.ColorJitter(
                brightness=np.random.rand(), 
                contrast=5 * np.random.rand(),
                hue=np.random.rand() / 2, 
                saturation=np.random.rand()
            )
            image = color_aug(image)
            image = np.array(image)
        
        # Normalize image
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-9)
        image = np.transpose(image, [2, 0, 1])
        
        # Create lower resolution image for dual-branch mode
        lower_size = 56
        lower_image = cv2.resize(np.transpose(image, [1, 2, 0]), (lower_size, lower_size))
        lower_image = (lower_image - np.min(lower_image)) / (np.max(lower_image) - np.min(lower_image) + 1e-9)
        lower_image = np.transpose(lower_image, [2, 0, 1])
        
        # Default SMPL parameters (zeros if no annotations)
        pose = np.zeros((24, 3, 3), dtype=np.float32)
        betas = np.zeros(10, dtype=np.float32)
        keypoints_3d = np.zeros((24, 3), dtype=np.float32)
        
        # Load annotations if available
        image_filename = os.path.basename(image_path)
        if self.has_annotations and image_filename in self.annotations:
            ann = self.annotations[image_filename]
            
            # Extract SMPL parameters if available
            if 'pose' in ann:
                pose = np.array(ann['pose']).reshape(24, 3, 3)
            if 'betas' in ann:
                betas = np.array(ann['betas']).reshape(10)
            if 'keypoints_3d' in ann:
                keypoints_3d = np.array(ann['keypoints_3d']).reshape(-1, 3)
        
        # Create SMPL parameter dictionaries
        smpl_params = {
            'global_orient': pose[0, :, :],
            'body_pose': pose[1:, :, :],
            'betas': betas
        }
        
        has_smpl_params = {
            'global_orient': self.has_body_pose[index],
            'body_pose': self.has_body_pose[index],
            'betas': self.has_betas[index]
        }
        
        smpl_params_is_axis_angle = {
            'global_orient': False,
            'body_pose': False,
            'betas': False
        }
        
        # Create item dictionary
        item = {
            'img': image,
            'img_lower': lower_image,
            'img_size': float(self.image_size),
            'smpl_params': smpl_params,
            'has_smpl_params': has_smpl_params,
            'smpl_params_is_axis_angle': smpl_params_is_axis_angle,
            'keypoints_3d': keypoints_3d,
            'imgroot': image_path,
            'idx': index,
            'flag': np.ones(9, dtype=np.float32)  # Default flag for dual-branch
        }
        
        return item
    
    def __len__(self):
        return self.length


class SingleBranchSMPLDiffusion:
    """
    Modified SMPLDiffusion that works with single-branch features only
    """
    
    def __init__(self, args, cfg):
        if SMPLDiffusion is None:
            raise ImportError("SMPLDiffusion not available - missing dependencies")
        
        # Create instance of the original class
        self._original = SMPLDiffusion(args, cfg)
        
        # Copy attributes
        for attr in dir(self._original):
            if not attr.startswith('_') and not callable(getattr(self._original, attr)):
                setattr(self, attr, getattr(self._original, attr))
        
        # Copy important methods
        self.T = self._original.T
        self.npose = self._original.npose
        self.cfg = self._original.cfg
        self.p_mean_variance = self._original.p_mean_variance
        if hasattr(self._original, 'fc_head'):
            self.fc_head = self._original.fc_head
        
    def forward(self, x_T, feats, flag=None):
        """
        Single-branch forward pass
        Args:
            x_T: Input noise tensor
            feats: Features from main backbone only
            flag: Optional flag (not used in single-branch mode)
        """
        batch_size = feats.shape[0]
        
        # Use only the main model (no zoom branch)
        for tdx in reversed(range(self.T)):
            t = x_T.new_ones([x_T.shape[0], ], dtype=torch.long) * tdx
            mean, var = self.p_mean_variance(x_t=x_T, t=t, feats=feats)
            if tdx > 0:
                noise = torch.randn_like(x_T)
            else:
                noise = 0
            x_T = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_T).int().sum() == 0, "nan in tensor."
        
        # Get final predictions
        x_0 = x_T
        pred_pose = x_0[:, :self.npose]
        pred_betas = self.fc_head_single(feats)  # Use single-branch head
        
        # Convert to rotation matrices
        if rot6d_to_rotmat is not None:
            pred_pose_output = rot6d_to_rotmat(pred_pose.reshape(batch_size, -1)).view(
                batch_size, self.cfg.SMPL.NUM_BODY_JOINTS + 1, 3, 3)
        else:
            # Fallback: assume pose is already in proper format
            pred_pose_output = pred_pose.view(batch_size, -1, 3, 3)
        
        pred_smpl_params = {
            'global_orient': pred_pose_output[:, [0]],
            'body_pose': pred_pose_output[:, 1:]
        }
        pred_smpl_params['betas'] = pred_betas.view(-1, 10)
        
        return pred_smpl_params, None, x_T, pred_pose, None
    
    def fc_head_single(self, feats):
        """Single-branch feature head for beta prediction"""
        # Use the existing fc_head but only with main features
        if hasattr(self, 'fc_head'):
            # Create dummy zoom features of zeros
            dummy_feats_zoom = torch.zeros_like(feats)
            return self.fc_head(feats, dummy_feats_zoom)
        else:
            # Fallback: create a simple linear layer if fc_head doesn't exist
            if not hasattr(self, '_single_head'):
                feat_dim = feats.shape[1]
                self._single_head = torch.nn.Linear(feat_dim, 10).to(feats.device)
            return self._single_head(feats.mean(dim=(2, 3)))  # Global average pooling


def parse_config():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EgoHMR Custom Evaluation')
    
    # Existing arguments
    parser.add_argument('--BENCHMARK', default=True, type=bool)
    parser.add_argument('--DETERMINISTIC', default=False, type=bool)
    parser.add_argument('--ENABLED', default=True, type=bool)
    parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file')
    parser.add_argument('--gpu', default=0, help='GPU to use', type=int)
    parser.add_argument('--test_batch_size', default=8, help='batch-size for testing', type=int)
    parser.add_argument('--num_workers', default=4, help='number of workers', type=int)
    
    # Custom evaluation arguments
    parser.add_argument('--single_branch', action='store_true', help='Use single branch mode')
    parser.add_argument('--custom_dataset_path', type=str, default='.',
                       help='Path to custom image directory')
    parser.add_argument('--custom_annotations', type=str, default=None,
                       help='Path to annotations file (JSON/MAT/pickle)')
    parser.add_argument('--annotation_format', type=str, default='auto',
                       choices=['json', 'mat', 'pickle', 'auto'],
                       help='Format of annotation file')
    parser.add_argument('--output_path', type=str, default='./custom_results',
                       help='Path to save results')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_size', type=int, default=384,
                       help='Input image size')
    
    # Diffusion model arguments  
    parser.add_argument('--T', default=10, help='T steps for diffusion model', type=int)
    parser.add_argument('--beta_l', default=1e-4, help='starting beta value', type=float)
    parser.add_argument('--beta_T', default=0.02, help='ending beta value', type=float)
    
    return parser.parse_args()


def test_custom_dataset(args, LOGGER=None):
    """
    Evaluate model on custom dataset
    """
    if LOGGER is None:
        LOGGER = ConsoleLogger('custom_evaluation', 'test')
    
    LOGGER.info("Starting custom dataset evaluation")
    LOGGER.info(f"Single branch mode: {args.single_branch}")
    LOGGER.info(f"Dataset path: {args.custom_dataset_path}")
    LOGGER.info(f"Output path: {args.output_path}")
    
    # Setup CUDA
    cudnn.benchmark = args.BENCHMARK
    cudnn.deterministic = args.DETERMINISTIC
    cudnn.enabled = args.ENABLED
    
    # Load model config
    if args.model_cfg is None:
        model_cfg = prohmr_config()
    else:
        model_cfg = get_config(args.model_cfg)
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Create custom dataset
    test_dataset = CustomDataset(
        image_dir=args.custom_dataset_path,
        annotations_path=args.custom_annotations,
        image_size=args.image_size,
        stage='Test',
        annotation_format=args.annotation_format
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=args.num_workers
    )
    
    LOGGER.info(f"Loaded {len(test_dataset)} samples")
    
    # Create models
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Feature extractors
    feature_bone = create_backbone(model_cfg).to(device)
    
    if args.single_branch:
        # Single branch mode
        diff_bone = SingleBranchSMPLDiffusion(args, model_cfg).to(device)
        feature_bone_zoom = None
    else:
        # Dual branch mode  
        feature_bone_zoom = create_backbone(model_cfg).to(device)
        diff_bone = SMPLDiffusion(args, model_cfg).to(device)
    
    smpl_bone = SMPLHead().to(device)
    
    # Load checkpoint
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"No checkpoint found at {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    feature_bone.load_state_dict(checkpoint['feature_bone_state_dict'])
    
    if not args.single_branch and feature_bone_zoom is not None:
        feature_bone_zoom.load_state_dict(checkpoint['feature_bone_zoom_state_dict'])
    
    diff_bone.load_state_dict(checkpoint['diff_bone_state_dict'])
    LOGGER.info('Finished loading models')
    
    # Evaluation
    feature_bone.eval()
    if feature_bone_zoom is not None:
        feature_bone_zoom.eval()
    diff_bone.eval()
    
    # Storage for results
    pred_keypoints_3d_list = []
    gt_keypoints_3d_list = []
    pred_vertices_list = []
    image_paths_list = []
    
    # Initialize evaluator if we have ground truth
    evaluator = None
    if test_dataset.has_annotations:
        # Standard joint indices for evaluation
        keypoint_list = list(range(24))  # Use all SMPL joints
        pelvis_ind = 0  # SMPL pelvis index
        evaluator = Evaluator(
            dataset_length=len(test_dataset),
            keypoint_list=keypoint_list,
            pelvis_ind=pelvis_ind,
            metrics=['mode_mpjpe', 'mode_re']
        )
    
    LOGGER.info('Starting inference')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            batch_size = batch['img'].shape[0]
            
            # Move to device
            x = batch['img'].to(torch.float32).to(device)
            
            # Extract features
            conditioning_feats = feature_bone(x)
            
            if args.single_branch:
                # Single branch inference
                params_noise = torch.randn(batch_size, diff_bone.npose, device=device)
                pred_smpl_params, _, _, _, _ = diff_bone(params_noise, conditioning_feats)
            else:
                # Dual branch inference
                x_zoom = batch['img_lower'].to(torch.float32).to(device)
                flag = batch['flag'].to(torch.float32).to(device)
                conditioning_feats_zoom = feature_bone_zoom(x_zoom)
                
                params_noise = torch.randn(batch_size, diff_bone.npose, device=device)
                params_noise_zoom = torch.randn(batch_size, diff_bone.npose_lower, device=device)
                
                pred_smpl_params, _, _, _, _ = diff_bone(
                    params_noise, params_noise_zoom, 
                    conditioning_feats, conditioning_feats_zoom, flag
                )
            
            # Get SMPL output
            pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
            pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
            pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
            
            smpl_output, _ = smpl_bone(
                global_orient=pred_smpl_params['global_orient'],
                body_pose=pred_smpl_params['body_pose'], 
                betas=pred_smpl_params['betas']
            )
            
            pred_keypoints_3d = smpl_output.joints.cpu().numpy()
            pred_vertices = smpl_output.vertices.cpu().numpy()
            
            # Store results
            pred_keypoints_3d_list.append(pred_keypoints_3d)
            pred_vertices_list.append(pred_vertices)
            
            # Store ground truth and image paths
            gt_keypoints_3d = batch['keypoints_3d'].cpu().numpy()
            gt_keypoints_3d_list.append(gt_keypoints_3d)
            image_paths_list.extend(batch['imgroot'])
            
            # Update evaluator if available
            if evaluator is not None and test_dataset.has_annotations:
                # Create output dict for evaluator
                output = {
                    'pred_keypoints_3d': torch.from_numpy(pred_keypoints_3d).unsqueeze(1)  # Add sample dimension
                }
                batch_eval = {
                    'keypoints_3d': batch['keypoints_3d'].unsqueeze(1)  # Add sample dimension
                }
                evaluator(output, batch_eval)
    
    # Concatenate all results
    pred_keypoints_3d_all = np.concatenate(pred_keypoints_3d_list, axis=0)
    pred_vertices_all = np.concatenate(pred_vertices_list, axis=0)
    gt_keypoints_3d_all = np.concatenate(gt_keypoints_3d_list, axis=0)
    
    # Save results
    results = {
        'pred_keypoints_3d': pred_keypoints_3d_all,
        'pred_vertices': pred_vertices_all,
        'gt_keypoints_3d': gt_keypoints_3d_all,
        'image_paths': image_paths_list,
        'config': {
            'single_branch': args.single_branch,
            'image_size': args.image_size,
            'model_path': args.model_path,
            'dataset_path': args.custom_dataset_path
        }
    }
    
    # Save in multiple formats
    # NumPy format
    np.save(os.path.join(args.output_path, 'results.npy'), results)
    
    # MAT format
    scipy.io.savemat(os.path.join(args.output_path, 'results.mat'), {
        'pred_keypoints_3d': pred_keypoints_3d_all,
        'pred_vertices': pred_vertices_all,
        'gt_keypoints_3d': gt_keypoints_3d_all,
        'image_paths': image_paths_list
    })
    
    # JSON format (without large arrays)
    with open(os.path.join(args.output_path, 'config.json'), 'w') as f:
        json.dump(results['config'], f, indent=2)
    
    # Print evaluation metrics
    if evaluator is not None and test_dataset.has_annotations:
        LOGGER.info("Evaluation Metrics:")
        evaluator.log()
        
        # Save metrics
        metrics = {}
        for metric in evaluator.metrics:
            if hasattr(evaluator, metric):
                metric_values = getattr(evaluator, metric)[:evaluator.counter]
                metrics[metric] = {
                    'mean': float(metric_values.mean()),
                    'std': float(metric_values.std()),
                    'values': metric_values.tolist()
                }
        
        with open(os.path.join(args.output_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    else:
        LOGGER.info("No ground truth available - skipping evaluation metrics")
    
    LOGGER.info(f"Results saved to {args.output_path}")
    LOGGER.info("Custom dataset evaluation completed")


def test_basic_functionality():
    """Test basic functionality without model loading"""
    print("Testing basic functionality...")
    
    # Test CustomDataset creation
    try:
        # Create a dummy image directory for testing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy image
            dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            dummy_image_path = os.path.join(temp_dir, "test_image.jpg")
            
            from PIL import Image
            Image.fromarray(dummy_image).save(dummy_image_path)
            
            # Test dataset creation
            dataset = CustomDataset(
                image_dir=temp_dir,
                annotations_path=None,
                image_size=224,
                stage='Test'
            )
            
            print(f"Dataset created successfully with {len(dataset)} samples")
            
            # Test getting an item
            if len(dataset) > 0:
                item = dataset[0]
                print(f"Item keys: {list(item.keys())}")
                print(f"Image shape: {item['img'].shape}")
                print(f"Lower image shape: {item['img_lower'].shape}")
                
        print("Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"Basic functionality test failed: {e}")
        return False


def main():
    """Main function"""
    args = parse_config()
    
    # If no model path is provided, run basic test
    if args.model_path == 'test':
        test_basic_functionality()
        return
        
    test_custom_dataset(args)


if __name__ == '__main__':
    main()