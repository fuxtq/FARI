from fari import FARIEditor


editor = FARIEditor.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    device="cuda",
    cpu_offload=True,
)
result = editor.edit(
    source_prompt="A white cup on a wooden table.",
    target_prompt="A blue cup on a wooden table.",
    family="change_color",
    preserve_mask="path/to/preserve_mask.png",
    seed=42,
)
result.save("outputs/cup_color")
