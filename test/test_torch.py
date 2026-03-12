import logging
import tempfile
import traceback

import paddle

from les import Les

les = Les(les_arguments={})
paddle.manual_seed(0)
r = paddle.rand(10, 3) * 10
r.requires_grad_(requires_grad=True)
q = paddle.rand(10) * 2 - 1
box_full = paddle.tensor([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
result = les(
    desc=r, positions=r, cell=box_full.unsqueeze(0), batch=None, compute_bec=False
)
for name, module in les.named_modules():
    try:
        scripted_module = paddle.jit.to_static(function=module)
        with tempfile.NamedTemporaryFile() as tmp:
            paddle.jit.save(layer=scripted_module, path=tmp.name.rsplit(".", 1)[0])
    except Exception as e:
        print(f"Save failed: {name}, Error: {e}")
try:
    scripted_model = paddle.jit.to_static(function=les)
    with tempfile.NamedTemporaryFile() as tmp:
        paddle.jit.save(layer=scripted_model, path=tmp.name.rsplit(".", 1)[0])
    print("Model scripted and saved successfully.")
except Exception as e:
    logging.error(f"Error scripting or saving the model: {e}")
    logging.error(traceback.format_exc())
try:
    for i in range(3):
        script_result = scripted_model(
            desc=r,
            positions=r,
            cell=box_full.unsqueeze(0),
            batch=None,
            compute_bec=False,
        )
except Exception as e:
    logging.error(f"Error: {e}")
    logging.error(traceback.format_exc())
if result.keys() != script_result.keys():
    print("Keys do not match")
else:
    for k in result.keys():
        if result[k] is not None and paddle.compat.allclose(
            result[k], script_result[k]
        ):
            print(f"Key: {k} \n Torchscript result is identical to original result.")
les_no_atomwise = Les(les_arguments={"use_atomwise": False})
paddle.jit.to_static(function=les_no_atomwise)
result = les(
    latent_charges=q,
    positions=r,
    cell=box_full.unsqueeze(0),
    batch=None,
    compute_bec=False,
)
