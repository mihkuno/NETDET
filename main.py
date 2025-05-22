from inference import predict_single_sample, load_model_components

# Example: predict a single sample string
# sample_input = "0,icmp,eco_i,SF,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,65,0.00,0.00,0.00,0.00,1.00,0.00,1.00,3,57,1.00,0.00,1.00,0.28,0.00,0.00,0.00,0.00" # this is 'saint' not found in train
sample_input = "0,tcp,private,REJ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,136,1,0.00,0.00,1.00,1.00,0.01,0.06,0.00,255,1,0.00,0.06,0.00,0.00,0.00,0.00,1.00,1.00" # this is neptuen


args = {
    'model_path': 'output/nsl_model.pth',  # Update with your model path
}

model, encoders, scaler, device = load_model_components(args['model_path'])

pred_label, pred_prob = predict_single_sample(sample_input, model, encoders, scaler, device)
print(f"\nSingle sample prediction:\nLabel: {pred_label} | Confidence: {pred_prob*100:.2f}%")
