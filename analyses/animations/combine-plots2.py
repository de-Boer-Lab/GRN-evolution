from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import os

# --- helpers --------------------------------------------------------------

def open_rgb(path):
    return Image.open(path).convert("RGB")

def resize_to_height(img, target_h):
    w, h = img.size
    if h == target_h:
        return img
    new_w = int(round(w * (target_h / h)))
    return img.resize((new_w, target_h), Image.LANCZOS)

def pad_to_width(img, target_w, fill=(255, 255, 255)):
    w, h = img.size
    if w == target_w:
        return img
    new = Image.new("RGB", (target_w, h), fill)
    x = (target_w - w) // 2
    new.paste(img, (x, 0))
    return new

def add_left_ribbon_outside(img, color, stripe_w=8):
    """
    Add a colored ribbon to the LEFT *outside* the image
    (increases width; does not cover the GRN).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    out = Image.new("RGB", (w + stripe_w, h), (255, 255, 255))
    # ribbon
    draw = ImageDraw.Draw(out)
    draw.rectangle([(0, 0), (stripe_w, h)], fill=color)
    # paste original next to ribbon
    out.paste(img, (stripe_w, 0))
    return out

def compose_grid_2x2(
    tl_path, bl_path, tr_path, br_path,
    out_path,
    tl_color, bl_color, 
    col_gutter=16, row_gutter=16,  # mpl blue/orange
    stripe_w=8
):
    """
    2x2 grid:
      TL: recomb GRN   (thin left stripe + small corner tag)
      BL: no-recomb GRN
      TR: fitness
      BR: complexity
    """
    # Load
    TL = open_rgb(tl_path)
    BL = open_rgb(bl_path)
    TR = open_rgb(tr_path)
    BR = open_rgb(br_path)

    # Normalize heights using right column as reference
    target_h = TR.height
    BR = resize_to_height(BR, target_h)
    TL = resize_to_height(TL, target_h)
    BL = resize_to_height(BL, target_h)

    # Add subtle cues on left tiles
    # Add subtle cues on left tiles
    TL = add_left_ribbon_outside(TL, tl_color, stripe_w=stripe_w)
    BL = add_left_ribbon_outside(BL, bl_color, stripe_w=stripe_w)

    # Equalize widths within columns
    left_w  = max(TL.width, BL.width)
    right_w = max(TR.width, BR.width)
    TL = pad_to_width(TL, left_w)
    BL = pad_to_width(BL, left_w)
    TR = pad_to_width(TR, right_w)
    BR = pad_to_width(BR, right_w)

    # Canvas
    tile_h  = target_h
    total_w = left_w + col_gutter + right_w
    total_h = tile_h + row_gutter + tile_h
    canvas = Image.new("RGB", (total_w, total_h), "white")

    # Paste
    canvas.paste(TL, (0, 0))
    canvas.paste(BL, (0, tile_h + row_gutter))
    canvas.paste(TR, (left_w + col_gutter, 0))
    canvas.paste(BR, (left_w + col_gutter, tile_h + row_gutter))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
    return out_path

# --- driver loop ----------------------------------------------------------

# Generation stepping logic matches your previous script
start, end = 1, 100000
generation = start

# Paths (adjust these to your actual filenames)
# Top-left (recomb) and bottom-left (no recomb) GRN images:
#   Assuming "sexual_*" = recomb and "asexual_*" = no recomb
# Right column plot frames produced earlier:
#   Fitness:  for-submission/frames_fitness/generation{g}.fitness.png
#   Complexity: for-submission/frames_complexity/generation{g}.complexity.png

while generation <= end:
    print(generation)

    # build paths for this generation
    recomb_grn_path    = f'for-submission/testH0generation{generation}.png'   # TL
    norecomb_grn_path  = f'for-submission/testF0generation{generation}.png'   # BL
    fitness_image_path    = f'for-submission/frames_fitness/generation{generation}.fitness.png'       # TR
    complexity_image_path = f'for-submission/frames_complexity/generation{generation}.complexity.png' # BR

    out_path = os.path.join('for-submission/combined_grid', f'generation{generation}.png')
    compose_grid_2x2(
        tl_path=recomb_grn_path, bl_path=norecomb_grn_path,
        tr_path=fitness_image_path, br_path=complexity_image_path,
        out_path=out_path,
        col_gutter=24, row_gutter=24,
        bl_color='#ff7f0e', tl_color='#2ca02c',  # or switch to green: (44,160,44)
        stripe_w=8
    )


    # step gens
    if generation < 500:
        generation += 1
    elif generation < 2000:
        generation += 10
    else:
        generation += 50
        


# from PIL import Image, ImageChops
# import os

# def crop_fixed(img, top=0, right=0, bottom=0, left=0):
#     w, h = img.size
#     return img.crop((left, top, w - right, h - bottom))

# def pad_to_height(img, target_h, align='center', fill=(255, 255, 255)):
#     w, h = img.size
#     if h == target_h:
#         return img
#     if h > target_h:
#         return img.resize((w, target_h), Image.LANCZOS)  # or raise if you never want to shrink
#     # pad smaller image
#     delta = target_h - h
#     if align == 'top':
#         top_pad, bot_pad = 0, delta
#     elif align == 'bottom':
#         top_pad, bot_pad = delta, 0
#     else:  # center
#         top_pad, bot_pad = delta // 2, delta - delta // 2
#     new = Image.new("RGB" if img.mode == "RGB" else "RGBA", (w, target_h), fill)
#     new.paste(img, (0, top_pad))
#     return new

# def stack_images_vertically(top_image_path, bottom_image_path, crop_top_px=0, crop_bottom_px=0):
#     top_img = Image.open(top_image_path)
#     bot_img = Image.open(bottom_image_path)
#     # optional trimming of fixed headers/footers on EACH image before stacking
#     top_img = crop_fixed(top_img, top=crop_top_px, bottom=crop_bottom_px)
#     bot_img = crop_fixed(bot_img, top=crop_top_px, bottom=crop_bottom_px)

#     if top_img.width != bot_img.width:
#         # pad narrower image to match width (rare) instead of erroring
#         target_w = max(top_img.width, bot_img.width)
#         def pad_w(img):
#             if img.width == target_w:
#                 return img
#             new = Image.new(img.mode, (target_w, img.height), (255, 255, 255))
#             new.paste(img, ((target_w - img.width)//2, 0))
#             return new
#         top_img, bot_img = pad_w(top_img), pad_w(bot_img)

#     total_h = top_img.height + bot_img.height
#     stacked = Image.new("RGBA", (top_img.width, total_h), (255, 255, 255, 0))
#     stacked.paste(top_img, (0, 0))
#     stacked.paste(bot_img, (0, top_img.height))
#     return stacked

# def resize_to_match_height(img, target_h):
#     w, h = img.size
#     new_w = int(w * target_h / h)
#     return img.resize((new_w, target_h), Image.LANCZOS)

# def combine_images(
#     top_image_path, bottom_image_path, fitness_image_path,
#     output_directory, output_filename,
#     gutter_px=0
# ):
#     # open and stack left column
#     top_img = Image.open(top_image_path).convert("RGB")
#     bot_img = Image.open(bottom_image_path).convert("RGB")
#     grn_stack = Image.new("RGB", (max(top_img.width, bot_img.width), top_img.height + bot_img.height), "white")
#     grn_stack.paste(top_img, (0, 0))
#     grn_stack.paste(bot_img, (0, top_img.height))

#     # open fitness panel
#     fitness = Image.open(fitness_image_path).convert("RGB")

#     # scale left stack to match right panel height
#     grn_stack = resize_to_match_height(grn_stack, fitness.height)

#     # new canvas
#     total_w = grn_stack.width + gutter_px + fitness.width
#     total_h = fitness.height
#     canvas = Image.new("RGB", (total_w, total_h), "white")

#     # paste both
#     canvas.paste(grn_stack, (0, 0))
#     canvas.paste(fitness, (grn_stack.width + gutter_px, 0))

#     os.makedirs(output_directory, exist_ok=True)
#     outpath = os.path.join(output_directory, output_filename)
#     canvas.save(outpath, quality=95)
#     return outpath


# start, end = 0, 100000
# generation = start

# while generation <= end:
#     print(generation)
#     if generation < 500:
#         generation += 1
#     elif generation < 2000:
#         generation += 10
#     else:
#         generation += 50

#     asexual_binding_affinity_image = 'for-submission/testF0generation{0}.png'.format(generation)
#     sexual_binding_affinity_image = 'for-submission/testH0generation{0}.png'.format(generation)
#     # asexual_affExp_image = 'checkpoints/changing-environments/testF/heatmaps/generation{0}-AffExp.png'.format(generation)
#     # sexual_affExp_image = 'checkpoints/testD/generation{0}-AffExp.png'.format(generation)
    
#     # top_image = stack_images_horizontally(asexual_binding_affinity_image, asexual_affExp_image)
#     # bottom_image = stack_images_horizontally(sexual_binding_affinity_image, sexual_affExp_image)
#     fitness_image = 'for-submission/fitness/generation{0}.png'.format(generation)
#     output_directory = 'for-submission/combined/'
#     output_filename = 'generation{0}.png'.format(generation)
    
#     # full_image = triple_stack(sexual_binding_affinity_image, sexual_affExp_image, fitness_image)
#     # full_image = combine_images(sexual_binding_affinity_image, sexual_affExp_image, fitness_image, output_directory, output_filename)
    
    
#     # top_image = Image.open(asexual_binding_affinity_image)
#     # bottom_image = Image.open(sexual_binding_affinity_image)
    
#     full_image = combine_images(asexual_binding_affinity_image, sexual_binding_affinity_image, fitness_image, output_directory, output_filename)
#     output_path = os.path.join(output_directory, output_filename)