import os
import random
import logging
import uuid
import tempfile
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
from google.cloud import storage

app = Flask(__name__, static_folder="static")
POSTER_SIZE = (1200, 1500)  # 4:5 ratio

# --- GLOBAL CONTROLS ---
MAX_CHAR_WIDTH = 400
MAX_CHAR_HEIGHT = 600
MAX_LOGO_WIDTH = 240
MAX_LOGO_HEIGHT = 240

GCS_BUCKET        = "marketing-instagram-posters"
GCS_PARENT_FOLDER = "instagram-image-posts"

def upload_file_to_gcs(bucket: str, parent: str, batch: str,
                       src_local_path: str, dst_name: str) -> str:
    client = storage.Client()
    blob_path = f"{parent}/{batch}/{dst_name}"
    blob = client.bucket(bucket).blob(blob_path)
    logging.info("Uploading %s → gs://%s/%s", src_local_path, bucket, blob_path)
    blob.upload_from_filename(src_local_path)
    return f"https://storage.googleapis.com/{bucket}/{blob_path}"

def hex_to_rgba(hex_str, alpha=255):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join([c*2 for c in hex_str])
    rgb = tuple(int(hex_str[i:i+2], 16) for i in (0,2,4))
    return rgb + (alpha,)

def wrap_text(text, font, max_width):
    words, lines, current = text.split(), [], ""
    for word in words:
        test = f"{current} {word}".strip()
        if font.getbbox(test)[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def parse_heading_to_lines(heading, default_color, font=None, max_width=None):
    """If font and max_width provided, wraps as one color. Otherwise, comma means new line."""
    if font and max_width:
        # Use word-wrapping
        return [[(s, default_color)] for s in wrap_text(heading, font, max_width)]
    # Else fallback to commas = line split (legacy)
    lines = []
    for line_part in heading.split(","):
        words = [(w, default_color) for w in line_part.strip().split()]
        if words:
            lines.append(words)
    return lines

# --- DATA ADAPTER ---
class PosterContext:
    def __init__(self, *,
        template_id,
        organization,
        heading,
        subheading="",
        background_color="#FFF",
        highlight_color="#1e1e1e"
    ):
        self.template_id = str(template_id).lower()
        self.organization = organization.lower()
        self.heading = heading or ""
        self.subheading = subheading or ""
        self.bg_color = background_color or "#FFF"
        self.highlight = highlight_color or "#1e1e1e"

# --- POSTER + CAROUSAL CLASSES ---

class BasePosterTemplate:
    output_size = POSTER_SIZE

    # Padding/margin and font can be customized per class
    LOGO_PADDING = (60, 40)
    LOGO_SIZE = (MAX_LOGO_WIDTH, MAX_LOGO_HEIGHT)

    def __init__(self, base_path, context: PosterContext):
        self.base = base_path
        self.ctx = context
        self.logo_path = os.path.join(self.base, "static", f"{self.ctx.organization}.png")
        if not os.path.exists(self.logo_path):
            raise FileNotFoundError(f"Logo '{self.logo_path}' not found.")

    def render_logo(self, bg):
        """Render logo at top left (default)."""
        icon = Image.open(self.logo_path).convert("RGBA")
        icon_w, icon_h = icon.size
        scale = min(self.LOGO_SIZE[0]/icon_w, self.LOGO_SIZE[1]/icon_h, 1.0)
        icon = icon.resize((int(icon_w*scale), int(icon_h*scale)), Image.LANCZOS)
        bg.paste(icon, self.LOGO_PADDING, icon)

class ClassicPosterTemplate(BasePosterTemplate):
    LEFT_MARGIN = 80
    HEADING_FONT_SIZE = 90
    SUBHEADING_FONT_SIZE = 54
    LOGO_SIZE = (MAX_LOGO_WIDTH, MAX_LOGO_HEIGHT)
    LOGO_PADDING = (60, 40)

    def generate(self, skip_character=False):
        ctx, size = self.ctx, self.output_size

        bg = Image.new("RGBA", size, hex_to_rgba(ctx.bg_color))
        self.render_logo(bg)

        # --- character
        if not skip_character:
            char_dir = os.path.join(self.base, "static", "characters", ctx.organization)
            char_imgs = [os.path.join(char_dir, x) for x in os.listdir(char_dir) if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not char_imgs: raise Exception(f"No character images found for {ctx.organization}")
            char_img = Image.open(random.choice(char_imgs)).convert("RGBA")
            orig_w, orig_h = char_img.size
            scale = min(MAX_CHAR_WIDTH/orig_w, MAX_CHAR_HEIGHT/orig_h, 1.0)
            new_w, new_h = int(orig_w*scale), int(orig_h*scale)
            png = char_img.resize((new_w, new_h), Image.LANCZOS)
            right_x = size[0] - new_w - 40
            bg.paste(png, (right_x, (size[1]-new_h)//2), png)

        font_main = ImageFont.truetype(os.path.join(self.base, "static", "GolosText-Bold.ttf"), self.HEADING_FONT_SIZE)
        font_sub = ImageFont.truetype(os.path.join(self.base, "static", "GolosText-Regular.ttf"), self.SUBHEADING_FONT_SIZE)
        right_x = size[0] - 40 - MAX_CHAR_WIDTH
        max_text_w = right_x - self.LEFT_MARGIN

        heading_lines = parse_heading_to_lines(ctx.heading, ctx.highlight, font=font_main, max_width=max_text_w)
        sub_lines = wrap_text(ctx.subheading, font_sub, max_text_w)
        block_h = (len(heading_lines)*(font_main.size+16)) + 54 + len(sub_lines)*font_sub.size
        start_y = (size[1] - block_h)//2
        draw = ImageDraw.Draw(bg)
        y = start_y
        for line in heading_lines:
            x = self.LEFT_MARGIN
            for word, color in line:
                draw.text((x, y), word, fill=hex_to_rgba(color), font=font_main)
                x += font_main.getbbox(word)[2]+20
            y += font_main.size+16
        y += 54
        for s in sub_lines:
            draw.text((self.LEFT_MARGIN, y), s, fill=hex_to_rgba(ctx.highlight), font=font_sub)
            y += font_sub.size
        return bg

class RightImageBackgroundPosterTemplate(ClassicPosterTemplate):
    """Just inherits classic; only background changes!"""
    def generate(self, skip_character=False):
        ctx, size = self.ctx, self.output_size
        bg_path = os.path.join(self.base, "static", "background", ctx.organization, f"{ctx.organization}-background.png")
        if os.path.exists(bg_path):
            bg = Image.open(bg_path).convert("RGBA").resize(size, Image.LANCZOS)
        else:
            bg = Image.new("RGBA", size, hex_to_rgba(ctx.bg_color))
        self.render_logo(bg)
        # character and text/drawing logic is as ClassicPosterTemplate
        self.output_size = size  # needed as classic expects it
        return super().generate(skip_character=skip_character)

class CarousalCenteredQuoteTemplate(BasePosterTemplate):
    # Flexible design controls up-front!
    TOP_MARGIN = 100
    BOTTOM_MARGIN = 100
    SIDE_MARGIN = 90

    MAIN_FONT_SIZE = 110
    MAIN_FONT_PATH = "GolosText-Bold.ttf"

    QUOTEMARK_FONT_PATH = "GolosText-Bold.ttf"
    QUOTEMARK_SIZE = 92
    QUOTEMARK_TOP = 56
    QUOTEMARK_COLOR = "#fff"

    LOGO_SIZE = 120
    LOGO_MARGIN_L = 40
    LOGO_MARGIN_B = 50

    BG_COLOR = "#35343A"  # Default carousal background

    def generate(self, skip_character=True):
        ctx, size = self.ctx, self.output_size
        bg = Image.new("RGBA", size, hex_to_rgba(ctx.bg_color if hasattr(ctx, "bg_color") else self.BG_COLOR))
        draw = ImageDraw.Draw(bg)

        # Quote mark at top center
        quote_font = ImageFont.truetype(os.path.join(self.base, "static", self.QUOTEMARK_FONT_PATH), self.QUOTEMARK_SIZE)
        quote = u"\u201C"
        quote_w = quote_font.getbbox(quote)[2]
        draw.text(((size[0] - quote_w) // 2, self.QUOTEMARK_TOP), quote, font=quote_font, fill=self.QUOTEMARK_COLOR)

        # Centered quote text block
        max_text_w = size[0] - 2 * self.SIDE_MARGIN
        main_font = ImageFont.truetype(os.path.join(self.base, "static", self.MAIN_FONT_PATH), self.MAIN_FONT_SIZE)
        lines = wrap_text(ctx.heading, main_font, max_text_w)

        line_height = main_font.getbbox("Ay")[3] - main_font.getbbox("Ay")[1]
        total_block_height = len(lines) * (line_height + 14)

        top_y = self.QUOTEMARK_TOP + self.QUOTEMARK_SIZE + 40
        usable_height = size[1] - top_y - self.BOTTOM_MARGIN - self.LOGO_SIZE - 40
        block_y = top_y + (usable_height - total_block_height) // 2

        y = block_y
        for line in lines:
            line_w = main_font.getbbox(line)[2]
            draw.text(((size[0] - line_w)//2, y), line, font=main_font, fill=hex_to_rgba(ctx.highlight))
            y += line_height + 14

        # Logo at bottom left
        icon = Image.open(self.logo_path).convert("RGBA")
        icon_w, icon_h = icon.size
        scale = min(self.LOGO_SIZE/icon_w, self.LOGO_SIZE/icon_h, 1.0)
        icon = icon.resize((int(icon_w*scale), int(icon_h*scale)), Image.LANCZOS)
        icon_y = size[1] - self.LOGO_MARGIN_B - icon.height
        bg.paste(icon, (self.LOGO_MARGIN_L, icon_y), icon)

        # Optionally org name beside logo
        # org_font = ImageFont.truetype(os.path.join(self.base, "static", "GolosText-Bold.ttf"), 40)
        # org_y = icon_y + (icon.height - 40)//2
        # draw.text((self.LOGO_MARGIN_L + icon.width + 22, org_y), ctx.organization, font=org_font, fill="#fff")
        return bg

class PosterTemplateFactory:
    templates = {
        'classic': ClassicPosterTemplate,
        '1': ClassicPosterTemplate,
        'rightimage': RightImageBackgroundPosterTemplate,
        '2': RightImageBackgroundPosterTemplate,
        'carousal_center': CarousalCenteredQuoteTemplate,
        'carousal': CarousalCenteredQuoteTemplate
    }
    @classmethod
    def get_template(cls, tid, *args, **kwargs):
        T = cls.templates.get(str(tid).lower())
        if not T: raise ValueError(f"Invalid poster_template_id: {tid}")
        return T(*args, **kwargs)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json(force=True)
        base = os.path.dirname(os.path.abspath(__file__))
        batch_id = uuid.uuid4().hex

        ctx = PosterContext(
            template_id=data.get("poster_template_id"),
            organization=data.get("organization"),
            heading=data.get("poster_heading", ""),
            subheading=data.get("poster_subheading", ""),
            background_color=data.get("poster_background_color", "#fff"),
            highlight_color=data.get("poster_highlight_color", "#1e1e1e")
        )
        img = PosterTemplateFactory.get_template(ctx.template_id, base, ctx).generate(skip_character=False)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            img.convert("RGB").save(tf.name, format="PNG")
            poster_url = upload_file_to_gcs(
                bucket=GCS_BUCKET,
                parent=GCS_PARENT_FOLDER,
                batch=batch_id,
                src_local_path=tf.name,
                dst_name=f"poster_{batch_id}.png"
            )
        os.remove(tf.name)

        carousal_urls = []
        for idx, citem in enumerate(data.get("carousal", [])):
            cctx = PosterContext(
                template_id=citem.get("carousal_template_id", "carousal_center"),
                organization=citem.get("organization", data.get("organization", "")),
                heading=citem.get("carousal_heading", ""),
                subheading=citem.get("carousal_subheading", ""),
                background_color=citem.get("carousal_background_color", "#35343A"),
                highlight_color=citem.get("carousal_highlight_color", "#fff")
            )
            template = PosterTemplateFactory.get_template(cctx.template_id, base, cctx)
            cimg = template.generate(skip_character=True)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as ctf:
                cimg.convert("RGB").save(ctf.name, format="PNG")
                url = upload_file_to_gcs(
                    bucket=GCS_BUCKET,
                    parent=GCS_PARENT_FOLDER,
                    batch=batch_id,
                    src_local_path=ctf.name,
                    dst_name=f"carousal_{idx+1}_{batch_id}.png"
                )
            os.remove(ctf.name)
            carousal_urls.append({"img_url": url})

        resp = {"img_url": poster_url}
        if carousal_urls:
            resp["carousal"] = carousal_urls

        return jsonify(resp)
    except FileNotFoundError as fnf:
        return jsonify({"error": str(fnf)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run('0.0.0.0', 8080)