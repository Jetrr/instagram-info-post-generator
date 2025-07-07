import os
import random
import logging
import uuid
import tempfile
import re
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
from google.cloud import storage

app = Flask(__name__, static_folder="static")
POSTER_SIZE = (1200, 1500)

MAX_CHAR_WIDTH = 300
MAX_CHAR_HEIGHT = 600
MAX_LOGO_WIDTH = 240
MAX_LOGO_HEIGHT = 240

GCS_BUCKET        = "marketing-instagram-posters"
GCS_PARENT_FOLDER = "instagram-image-posts"

ORG_WEBSITES = {
    "nerdii": "nerdii.co",
    "gaper": "gaper.io",
    # add more orgs if needed
}


def get_org_website(org):
    return ORG_WEBSITES.get(org.lower(), "")


def upload_file_to_gcs(bucket: str, parent: str, batch: str, src_local_path: str, dst_name: str) -> str:
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


def parse_colored_heading(heading, default_color, font=None, max_width=None):
    """
    Splits the heading into lines by comma, assigns color if [#abc123] specified before word.
    Performs wrapping if font+max_width provided.
    Returns: list of lines, line is list of (word, color).
    """
    result_lines = []
    for raw_line in heading.split(','):
        tokens = []
        for match in re.finditer(r'(?:\[(#(?:[A-Fa-f0-9]{3}|[A-Fa-f0-9]{6}))\])?([^\s,]+)', raw_line.strip()):
            color, word = match.groups()
            color = "#" + color if color else default_color
            tokens.append((word, color))
        if not font or not max_width:
            result_lines.append(tokens)
            continue
        # Wrapping handling
        linetokens = tokens[:]
        while linetokens:
            cur_line = []
            remaining = linetokens
            while remaining:
                test_words = [w for w, _ in cur_line + [remaining[0]]]
                if font.getbbox(' '.join(test_words))[2] <= max_width:
                    cur_line.append(remaining[0])
                    remaining = remaining[1:]
                else:
                    break
            if not cur_line:  # force at least one word per line to avoid infinite loop
                cur_line.append(linetokens[0])
                remaining = linetokens[1:]
            result_lines.append(cur_line)
            linetokens = remaining
    return result_lines


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


class BasePosterTemplate:
    output_size = POSTER_SIZE
    LOGO_PADDING = (60, 40)
    LOGO_SIZE = (MAX_LOGO_WIDTH, MAX_LOGO_HEIGHT)
    WEBSITE_FONT_SIZE = 37
    WEBSITE_MARGIN = (60, 35)
    SWIPE_ARROW_SIZE = 60
    SWIPE_ARROW_MARGIN = (40, 32)
    BG_COLOR = "#fff"

    def __init__(self, base_path, context: PosterContext):
        self.base = base_path
        self.ctx = context
        self.logo_path = os.path.join(self.base, "static", f"{self.ctx.organization}.png")
        if not os.path.exists(self.logo_path):
            raise FileNotFoundError(f"Logo '{self.logo_path}' not found.")

    def render_logo(self, bg):
        icon = Image.open(self.logo_path).convert("RGBA")
        icon_w, icon_h = icon.size
        scale = min(self.LOGO_SIZE[0] / icon_w, self.LOGO_SIZE[1] / icon_h, 1.0)
        icon = icon.resize((int(icon_w * scale), int(icon_h * scale)), Image.LANCZOS)
        bg.paste(icon, self.LOGO_PADDING, icon)
        return icon.height

    def render_website_bottom_left(self, bg):
        website = get_org_website(self.ctx.organization)
        if not website:
            return
        url_font_path = os.path.join(self.base, "static", "GolosText-Regular.ttf")
        try:
            url_font = ImageFont.truetype(url_font_path, self.WEBSITE_FONT_SIZE)
        except Exception:
            url_font = ImageFont.truetype("arial.ttf", self.WEBSITE_FONT_SIZE)
        url_w, url_h = url_font.getbbox(website)[2], url_font.getbbox(website)[3]
        margin_side, margin_bottom = self.WEBSITE_MARGIN
        pos = (margin_side, bg.size[1] - margin_bottom - url_h)
        draw = ImageDraw.Draw(bg)
        draw.text(pos, website, fill=hex_to_rgba("#666"), font=url_font)

    def render_swipe_arrow_bottom_right(self, bg):
        try:
            arrow_img = Image.open(os.path.join(self.base, "static", "swipe_arrow.png")).convert("RGBA")
            arrow_size = self.SWIPE_ARROW_SIZE
            arrow_img = arrow_img.resize((arrow_size, arrow_size), Image.LANCZOS)
            margin_r, margin_b = self.SWIPE_ARROW_MARGIN
            bg.paste(
                arrow_img,
                (bg.size[0] - margin_r - arrow_size, bg.size[1] - margin_b - arrow_size),
                arrow_img
            )
        except Exception as e:
            print("Swipe arrow could not be loaded:", e)


class ClassicPosterTemplate(BasePosterTemplate):
    LEFT_MARGIN = 80
    HEADING_FONT_SIZE = 90
    SUBHEADING_FONT_SIZE = 54
    BG_COLOR = "#fff"

    def generate(self, skip_character=False):
        ctx, size = self.ctx, self.output_size

        bg = Image.new("RGBA", size, hex_to_rgba(ctx.bg_color or self.BG_COLOR))
        self.render_logo(bg)

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

        heading_lines = parse_colored_heading(ctx.heading, ctx.highlight, font=font_main, max_width=max_text_w)
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

        self.render_website_bottom_left(bg)
        return bg


class CenterAlignedPosterTemplate(BasePosterTemplate):
    TOP_MARGIN = 60
    HEADING_FONT_SIZE = 90
    SUBHEADING_FONT_SIZE = 54
    LOGO_SIZE = (MAX_LOGO_WIDTH, MAX_LOGO_HEIGHT)
    LOGO_PADDING = (60, 40)
    SWIPE_ARROW_SIZE = 60

    def generate(self, skip_character=False):
        ctx, size = self.ctx, self.output_size
        bg_path = os.path.join(self.base, "static", "background", ctx.organization, f"{ctx.organization}-background.png")
        if os.path.exists(bg_path):
            bg = Image.open(bg_path).convert("RGBA").resize(size, Image.LANCZOS)
        else:
            bg = Image.new("RGBA", size, hex_to_rgba(ctx.bg_color or self.BG_COLOR ))
        self.render_logo(bg)

        font_main = ImageFont.truetype(os.path.join(self.base, "static", "GolosText-Bold.ttf"), self.HEADING_FONT_SIZE)
        font_sub = ImageFont.truetype(os.path.join(self.base, "static", "GolosText-Regular.ttf"), self.SUBHEADING_FONT_SIZE)
        max_text_w = size[0] - 2 * 110

        heading_lines = parse_colored_heading(ctx.heading, ctx.highlight, font=font_main, max_width=max_text_w)
        sub_lines = wrap_text(ctx.subheading, font_sub, max_text_w)

        line_height = font_main.getbbox("Ay")[3] - font_main.getbbox("Ay")[1]
        sub_line_height = font_sub.getbbox("Ay")[3] - font_sub.getbbox("Ay")[1]
        total_text_height = len(heading_lines) * (line_height + 10) + 35 + len(sub_lines) * (sub_line_height + 6)
        y = self.TOP_MARGIN + self.LOGO_SIZE[1] + 20

        draw = ImageDraw.Draw(bg)
        for line in heading_lines:
            x = (size[0] - font_main.getbbox(' '.join([w for w, _ in line]))[2]) // 2
            for idx, (word, color) in enumerate(line):
                draw.text((x, y), word, fill=hex_to_rgba(color), font=font_main)
                x += font_main.getbbox(word)[2] + (font_main.getbbox(" ")[2] if idx < len(line)-1 else 0)
            y += line_height + 10
        y += 25
        for s in sub_lines:
            xx = (size[0] - font_sub.getbbox(s)[2]) // 2
            draw.text((xx, y), s, fill=hex_to_rgba(ctx.highlight), font=font_sub)
            y += sub_line_height + 6

        if not skip_character:
            char_dir = os.path.join(self.base, "static", "characters", ctx.organization)
            char_imgs = [os.path.join(char_dir, x) for x in os.listdir(char_dir) if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not char_imgs: raise Exception(f"No character images found for {ctx.organization}")
            char_img = Image.open(random.choice(char_imgs)).convert("RGBA")
            orig_w, orig_h = char_img.size
            scale = min(MAX_CHAR_WIDTH/orig_w, MAX_CHAR_HEIGHT/orig_h, 1.0)
            new_w, new_h = int(orig_w*scale), int(orig_h*scale)
            png = char_img.resize((new_w, new_h), Image.LANCZOS)
            cx = (size[0] - new_w)//2
            cy = min(size[1] - new_h - 80, y + 30)
            bg.paste(png, (cx, cy), png)

        self.render_swipe_arrow_bottom_right(bg)
        self.render_website_bottom_left(bg)
        return bg


class CarousalCenteredQuoteTemplate(BasePosterTemplate):
    LEFT_MARGIN      = 80
    HEADING_FONT_SIZE   = 90
    LOGO_SIZE        = (MAX_LOGO_WIDTH, MAX_LOGO_HEIGHT)
    LOGO_PADDING     = (60, 40)
    WEBSITE_FONT_SIZE   = 37
    SWIPE_ARROW_SIZE = 60
    SIDE_MARGIN      = 80
    TOP_MARGIN       = 100
    BOTTOM_MARGIN    = 100
    BG_COLOR         = "#35343A"

    def render_logo(self, bg):
        icon = Image.open(self.logo_path).convert("RGBA")
        icon_w, icon_h = icon.size
        scale = min(self.LOGO_SIZE[0] / icon_w, self.LOGO_SIZE[1] / icon_h, 1.0)
        icon = icon.resize((int(icon_w * scale), int(icon_h * scale)), Image.LANCZOS)
        bg.paste(icon, self.LOGO_PADDING, icon)

    def generate(self, skip_character=True):
        ctx, size = self.ctx, self.output_size
        bg = Image.new("RGBA", size, hex_to_rgba(ctx.bg_color or self.BG_COLOR))
        draw = ImageDraw.Draw(bg)

        self.render_logo(bg)

        font_main = ImageFont.truetype(os.path.join(self.base, "static", "GolosText-Bold.ttf"), self.HEADING_FONT_SIZE)
        max_text_w = size[0] - 2 * self.SIDE_MARGIN
        heading_lines = parse_colored_heading(ctx.heading, ctx.highlight, font=font_main, max_width=max_text_w)
        line_height = font_main.getbbox("Ay")[3] - font_main.getbbox("Ay")[1]
        total_block_height = len(heading_lines) * (line_height + 16)
        y_start = (size[1] - total_block_height) // 2

        y = y_start
        for line in heading_lines:
            line_text = ' '.join(word for word, _ in line)
            line_w = font_main.getbbox(line_text)[2]
            x = (size[0] - line_w) // 2
            for idx, (word, color) in enumerate(line):
                draw.text((x, y), word, fill=hex_to_rgba(color), font=font_main)
                x += font_main.getbbox(word)[2] + (font_main.getbbox(" ")[2] if idx < len(line)-1 else 0)
            y += line_height + 16

        self.render_website_bottom_left(bg)
        self.render_swipe_arrow_bottom_right(bg)
        return bg


class PosterTemplateFactory:
    templates = {
        'p1': CenterAlignedPosterTemplate,
        'p2': ClassicPosterTemplate,
        'c1': CarousalCenteredQuoteTemplate,
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

        highlight_color = data.get("poster_font_color") or data.get("poster_highlight_color", "#1e1e1e")
        ctx = PosterContext(
            template_id=data.get("poster_template_id"),
            organization=data.get("organization"),
            heading=data.get("poster_heading", ""),
            subheading=data.get("poster_subheading", ""),
            background_color=data.get("poster_background_color", "#fff"),
            highlight_color=highlight_color
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
            carousal_highlight = citem.get("carousal_font_color") or citem.get("carousal_highlight_color") or "#fff"
            cctx = PosterContext(
                template_id=citem.get("carousal_template_id", "c1"),
                organization=citem.get("organization", data.get("organization", "")),
                heading=citem.get("carousal_heading", ""),
                subheading=citem.get("carousal_subheading", ""),
                background_color=citem.get("carousal_background_color", "#35343A"),
                highlight_color=carousal_highlight
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