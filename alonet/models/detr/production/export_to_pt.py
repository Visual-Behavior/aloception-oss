import torch
from alonet.detr import DetrR50

if __name__ == "__main__":
    device = torch.device("cuda")

    model = DetrR50(weights="detr-r50", tracing=True, aux_loss=False).eval()
    model.to(device)

    example_input = torch.rand(1, 4, 800, 1333).to(device)
    example_input2 = torch.rand(1, 4, 400, 600).to(device)

    traced_script_module = torch.jit.trace(model, example_input, check_inputs=[example_input, example_input2])
    traced_script_module.save("detr-r50.pt")

    print("[INFO] Model export successfully")
