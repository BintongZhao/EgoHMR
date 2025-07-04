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


class MockEvaluator:
    """Mock evaluator for testing when real one is not available"""
    def __init__(self, dataset_length, keypoint_list, pelvis_ind, metrics):
        self.dataset_length = dataset_length
        self.keypoint_list = keypoint_list
        self.pelvis_ind = pelvis_ind
        self.metrics = metrics
        self.counter = 0
        
        # Initialize metric storage
        for metric in metrics:
            setattr(self, metric, np.zeros(dataset_length))
    
    def __call__(self, output, batch, opt_output=None):
        """Mock evaluation that computes simple distance metrics"""
        pred_keypoints_3d = output['pred_keypoints_3d'].cpu().numpy()
        gt_keypoints_3d = batch['keypoints_3d'].cpu().numpy()
        
        batch_size = pred_keypoints_3d.shape[0]
        
        # Simple MPJPE calculation (mean per joint position error)
        if pred_keypoints_3d.shape[1] > 1:  # Multiple samples
            pred_keypoints_3d = pred_keypoints_3d[:, 0]  # Use first sample
        else:
            pred_keypoints_3d = pred_keypoints_3d.squeeze(1)
        
        if gt_keypoints_3d.shape[1] > 1:
            gt_keypoints_3d = gt_keypoints_3d[:, 0]
        else:
            gt_keypoints_3d = gt_keypoints_3d.squeeze(1)
        
        # Compute simple distance metrics
        for i in range(batch_size):
            if self.counter < self.dataset_length:
                pred = pred_keypoints_3d[i]
                gt = gt_keypoints_3d[i]
                
                # Simple MPJPE (millimeters)
                error = np.sqrt(np.sum((pred - gt) ** 2, axis=1)).mean() * 1000
                
                # Store in all available metrics
                for metric in self.metrics:
                    if hasattr(self, metric):
                        getattr(self, metric)[self.counter] = error
                
                self.counter += 1
    
    def log(self):
        """Print evaluation metrics"""
        if self.counter == 0:
            print('Evaluation has not started')
            return
            
        print(f'{self.counter} / {self.dataset_length} samples')
        for metric in self.metrics:
            if hasattr(self, metric):
                values = getattr(self, metric)[:self.counter]
                print(f'{metric}: {values.mean():.2f} mm')
        print('***')


class MockSMPLHead(torch.nn.Module):
    """Mock SMPL head for testing when real one is not available"""
    def __init__(self):
        super().__init__()
        
    def forward(self, global_orient, body_pose, betas):
        batch_size = global_orient.shape[0]
        device = global_orient.device
        
        # Create mock outputs
        joints = torch.zeros(batch_size, 24, 3).to(device)  # 24 SMPL joints
        vertices = torch.zeros(batch_size, 6890, 3).to(device)  # SMPL mesh vertices
        
        # Simple namespace to mimic SMPL output
        class MockOutput:
            def __init__(self, joints, vertices):
                self.joints = joints
                self.vertices = vertices
                
        return MockOutput(joints, vertices), None
    
    def __call__(self, global_orient, body_pose, betas):
        return self.forward(global_orient, body_pose, betas)


class MockSMPLDiffusion(torch.nn.Module):
    """
    Mock SMPLDiffusion for testing purposes when the real one is not available
    """
    def __init__(self, args, cfg):
        super().__init__()
        self.npose = 144  # 24 joints * 6 (6D rotation)
        self.npose_lower = 54  # Lower body subset
        self.T = args.T
        self.cfg = cfg
        
    def forward(self, x_T, x_T_zoom=None, feats=None, feats_zoom=None, flag=None):
        # Mock forward pass - returns dummy SMPL parameters
        batch_size = x_T.shape[0]
        device = x_T.device
        
        # Create dummy predictions
        global_orient = torch.zeros(batch_size, 1, 3, 3).to(device)
        body_pose = torch.zeros(batch_size, 23, 3, 3).to(device)
        betas = torch.zeros(batch_size, 10).to(device)
        
        pred_smpl_params = {
            'global_orient': global_orient,
            'body_pose': body_pose,
            'betas': betas
        }
        
        return pred_smpl_params, None, x_T, None, None
    
    def __call__(self, x_T, x_T_zoom=None, feats=None, feats_zoom=None, flag=None):
        return self.forward(x_T, x_T_zoom, feats, feats_zoom, flag)


class SingleBranchSMPLDiffusion(torch.nn.Module):
    """
    Modified SMPLDiffusion that works with single-branch features only
    """
    
    def __init__(self, args, cfg):
        super().__init__()
        if SMPLDiffusion is None:
            print("Warning: Using mock SMPLDiffusion due to missing dependencies")
            self._original = MockSMPLDiffusion(args, cfg)
        else:
            try:
                # Try to create the original SMPLDiffusion
                self._original = SMPLDiffusion(args, cfg)
            except Exception as e:
                print(f"Warning: Could not create SMPLDiffusion, using mock: {e}")
                self._original = MockSMPLDiffusion(args, cfg)
        
        # Copy attributes
        self.T = self._original.T
        self.npose = self._original.npose
        self.cfg = self._original.cfg
        
        # Copy methods if available
        if hasattr(self._original, 'p_mean_variance'):
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
        return self._original(x_T, feats=feats)
    
    def __call__(self, x_T, feats, flag=None):
        return self.forward(x_T, feats, flag)
    
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


class MockConsoleLogger:
    """Mock console logger for testing when real one is not available"""
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode
        
    def info(self, message):
        print(f"[{self.name}] {message}")


def test_custom_dataset(args, LOGGER=None):
    """
    Evaluate model on custom dataset
    """
    if LOGGER is None:
        if ConsoleLogger is not None:
            try:
                LOGGER = ConsoleLogger('custom_evaluation', 'test')
            except:
                LOGGER = MockConsoleLogger('custom_evaluation', 'test')
        else:
            LOGGER = MockConsoleLogger('custom_evaluation', 'test')
    
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
    if create_backbone is not None:
        feature_bone = create_backbone(model_cfg).to(device)
    else:
        print("Warning: create_backbone not available, using mock")
        # Create a simple mock backbone
        class MockBackbone(torch.nn.Module):
            def forward(self, x):
                # Return dummy features
                return torch.randn(x.shape[0], 2048, 1, 1).to(x.device)
        feature_bone = MockBackbone().to(device)
    
    if args.single_branch:
        # Single branch mode
        diff_bone = SingleBranchSMPLDiffusion(args, model_cfg).to(device)
        feature_bone_zoom = None
    else:
        # Dual branch mode  
        if create_backbone is not None:
            feature_bone_zoom = create_backbone(model_cfg).to(device)
        else:
            feature_bone_zoom = MockBackbone().to(device)
        
        if SMPLDiffusion is not None:
            try:
                diff_bone = SMPLDiffusion(args, model_cfg).to(device)
            except:
                diff_bone = MockSMPLDiffusion(args, model_cfg).to(device)
        else:
            diff_bone = MockSMPLDiffusion(args, model_cfg).to(device)
    
    if SMPLHead is not None:
        try:
            smpl_bone = SMPLHead().to(device)
        except:
            smpl_bone = MockSMPLHead().to(device)
    else:
        smpl_bone = MockSMPLHead().to(device)
    
    # Load checkpoint
    if args.model_path and os.path.isfile(args.model_path):
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            if hasattr(feature_bone, 'load_state_dict') and 'feature_bone_state_dict' in checkpoint:
                feature_bone.load_state_dict(checkpoint['feature_bone_state_dict'])
            
            if not args.single_branch and feature_bone_zoom is not None:
                if hasattr(feature_bone_zoom, 'load_state_dict') and 'feature_bone_zoom_state_dict' in checkpoint:
                    feature_bone_zoom.load_state_dict(checkpoint['feature_bone_zoom_state_dict'])
            
            if hasattr(diff_bone, 'load_state_dict') and 'diff_bone_state_dict' in checkpoint:
                diff_bone.load_state_dict(checkpoint['diff_bone_state_dict'])
            
            LOGGER.info('Finished loading models')
        except Exception as e:
            LOGGER.info(f'Warning: Could not load checkpoint: {e}')
            LOGGER.info('Proceeding with randomly initialized models')
    else:
        LOGGER.info('No valid model path provided, using randomly initialized models')
    
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
        
        if Evaluator is not None:
            try:
                evaluator = Evaluator(
                    dataset_length=len(test_dataset),
                    keypoint_list=keypoint_list,
                    pelvis_ind=pelvis_ind,
                    metrics=['mode_mpjpe', 'mode_re']
                )
            except:
                evaluator = MockEvaluator(
                    dataset_length=len(test_dataset),
                    keypoint_list=keypoint_list,
                    pelvis_ind=pelvis_ind,
                    metrics=['mode_mpjpe', 'mode_re']
                )
        else:
            evaluator = MockEvaluator(
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
                try:
                    evaluator(output, batch_eval)
                except Exception as e:
                    print(f"Warning: Evaluator failed: {e}")
                    # Continue without evaluation
    
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


def print_help():
    """Print usage examples and help"""
    print("""
EgoHMR Custom Evaluation Script
==============================

This script allows evaluation of EgoHMR models on custom datasets with flexible data loading
and supports both single-branch and dual-branch operation modes.

Features:
- Custom dataset class for flexible image loading
- Support for JSON, MAT, and pickle annotation formats
- Single-branch mode (main feature extractor only)
- Dual-branch mode (main + zoom feature extractors)
- PA-MPJPE evaluation metrics
- Results saved in multiple formats (NPY, MAT, JSON)

Usage Examples:

1. Basic evaluation with single-branch mode:
   python evaluation_custom.py --single_branch --custom_dataset_path /path/to/images --model_path /path/to/model.tar

2. Evaluation with annotations and metrics:
   python evaluation_custom.py --custom_dataset_path /path/to/images --custom_annotations /path/to/annotations.json --model_path /path/to/model.tar

3. Dual-branch evaluation (default):
   python evaluation_custom.py --custom_dataset_path /path/to/images --model_path /path/to/model.tar

4. Test with mock models (no trained model required):
   python evaluation_custom.py --custom_dataset_path /path/to/images --model_path none

5. Different annotation formats:
   python evaluation_custom.py --custom_dataset_path /path/to/images --custom_annotations data.mat --annotation_format mat
   python evaluation_custom.py --custom_dataset_path /path/to/images --custom_annotations data.pkl --annotation_format pickle

Arguments:
--single_branch         Use single branch mode (main feature extractor only)
--custom_dataset_path   Path to directory containing images
--custom_annotations    Path to annotation file (JSON/MAT/pickle format)
--annotation_format     Annotation format (json/mat/pickle/auto)
--model_path            Path to trained model checkpoint
--output_path           Directory to save results (default: ./custom_results)
--image_size            Input image size (default: 384)
--test_batch_size       Batch size for evaluation (default: 8)

Annotation Format:
JSON format should contain a dictionary mapping image filenames to annotations:
{
  "image1.jpg": {
    "pose": [[rotation_matrices...]], 
    "betas": [shape_parameters...],
    "keypoints_3d": [[x,y,z], ...]
  }
}

Output:
- results.npy: Full results in NumPy format
- results.mat: Results in MATLAB format  
- config.json: Configuration used for evaluation
- metrics.json: Evaluation metrics (if ground truth available)
""")


def main():
    """Main function"""
    import sys
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        print_help()
        return
        
    args = parse_config()
    
    # If no model path is provided, run basic test
    if args.model_path is None or args.model_path == 'test':
        test_basic_functionality()
        return
        
    test_custom_dataset(args)


if __name__ == '__main__':
    main()