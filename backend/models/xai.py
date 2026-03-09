
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class XAIGenerator:
    def __init__(self):
        pass

    def generate_heatmap(self, model, input_tensor, target_class_idx=None):
        """
        Generates Grad-CAM Heatmap.
        
        args:
            model: The pytorch model (DeepCareMTL)
            input_tensor: Preprocessed image tensor [1, 3, 224, 224]
        """
        # We need gradients, so we might need to re-run forward pass with gradients enabled
        # if the previous pass was no_grad.
        
        model.eval()
        # Enable gradients for this specific pass
        input_tensor.requires_grad_()
        
        # Hook for gradients
        gradients = []
        activations = []
        
        def save_gradient(grad):
            gradients.append(grad)
            
        def save_activation(module, input, output):
            activations.append(output)
            
        # Register hooks on the last conv layer
        target_layer = model.getLastConvLayer()
        
        handle_g = target_layer.register_full_backward_hook(lambda m, i, o: save_gradient(o[0]))
        handle_a = target_layer.register_forward_hook(save_activation)
        
        # Forward Pass
        a_logits, p_logits = model(input_tensor)
        
        # Zero grads
        model.zero_grad()
        
        # Target the Pathology Head
        if target_class_idx is None:
            target_class_idx = torch.argmax(p_logits)
            
        # Backward pass w.r.t the target class score
        score = p_logits[0, target_class_idx]
        score.backward()
        
        # Remove hooks
        handle_g.remove()
        handle_a.remove()
        
        if not gradients or not activations:
            return None, None
            
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
        
        # Weight the activations by the gradients
        # activations[0] is [1, 1280, 7, 7]
        activation = activations[0].detach()
        for i in range(activation.shape[1]):
            activation[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activation, dim=1).squeeze()
        
        # ReLU on heatmap
        heatmap = F.relu(heatmap)
        
        # Customize normalization to avoid div by zero
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.cpu().numpy()

    def overlay_heatmap(self, frame, heatmap):
        """
        Resize heatmap to frame size and overlay.
        """
        if heatmap is None:
            return frame, None
            
        # Resize heatmap to match frame
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        # Convert to uint8 0-255
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Superimpose
        superimposed = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
        
        return superimposed, heatmap_color
