from .text_base import TextBaseDataset
from ..smp import *


class ArcAGI(TextBaseDataset):

    TYPE = "VQA"
    DATASET_URL = {
        'ArcAGI1': 'https://opencompass.openxlab.space/utils/VLMEval/ArcAGI.tsv',
        'ArcAGI1-Image': 'https://opencompass.openxlab.space/utils/VLMEval/ArcAGI.tsv',
        'ArcAGI2': 'https://opencompass.openxlab.space/utils/VLMEval/ArcAGI2.tsv',
        'ArcAGI2-Image': 'https://opencompass.openxlab.space/utils/VLMEval/ArcAGI2.tsv'
    }
    DATASET_MD5 = {
        'ArcAGI1': '6e24018d4ff0f0c3d734ad65b0e852a9',
        'ArcAGI1-Image': '6e24018d4ff0f0c3d734ad65b0e852a9',
        'ArcAGI2': '1085690da452149e4d47f637f45b6a65',
        'ArcAGI2-Image': '1085690da452149e4d47f637f45b6a65',
    }
    SYSTEM_PROMPT_PREFIX = """\
You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Below is a list of input and output pairs with a pattern. \
Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, \
then apply that pattern to the test input to give a final output.

Respond in the format of the training output examples

--Training Examples--
"""
    SYSTEM_PROMPT_MID = """\
--End of Training Examples--

--Test Input--
"""
    SYSTEM_PROMPT_SUFFIX = """\
--End of Test Input--

Your response:
"""

    def __init__(self, dataset='ArcAGI2'):
        super().__init__(dataset)
        self.mode = 'text-only' if 'image' not in dataset.lower() else 'image-text'

    @staticmethod
    def generate_grid_image(
            grid,
            image_size=1000,
            bg_color="#FFFFFF",
            extra_bottom_padding_ratio=0,
            bordercol="white"):

        import drawsvg as dw
        import cairosvg

        min_pixel = 20
        max_pixel = 50
        cell_size = np.clip(image_size // len(grid), min_pixel, max_pixel)

        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)
        grid = grid.astype(int)
        assert np.all(grid >= 0) and np.all(grid <= 9), 'Grid values must be between 0 and 9.'
        padding = cell_size // 2
        cmap = [
            '#252525', '#0074D9', '#FF4136', '#37D449', '#FFDC00',
            '#808080', '#F012BE', '#FF871E', '#54D2EB', '#8D1D2C',
            '#FFFFFF'
        ]
        gridy, gridx = grid.shape

        grid_pixel_width = gridx * cell_size
        grid_pixel_height = gridy * cell_size
        extra_bottom_padding_px = cell_size * extra_bottom_padding_ratio

        total_image_width = grid_pixel_width + 2 * padding
        total_image_height = grid_pixel_height + 2 * padding + extra_bottom_padding_px

        line_thickness = 0.5
        border_width = 5
        lt = line_thickness / 2

        drawing = dw.Drawing(total_image_width, total_image_height, origin=(-padding, -padding))

        drawing.append(dw.Rectangle(
            -padding, -padding,
            total_image_width, total_image_height,
            fill=bg_color
        ))

        for j, row in enumerate(grid):
            for i, cell in enumerate(row):
                cell_index = int(cell) % len(cmap)
                drawing.append(dw.Rectangle(
                    i * cell_size + lt, j * cell_size + lt,
                    cell_size - lt, cell_size - lt, fill=cmap[cell_index]))

        drawing.append(dw.Rectangle(
            -padding, -padding, total_image_width, total_image_height,
            fill='none', stroke=bordercol, stroke_width=border_width))

        svg_string = drawing.as_svg()
        idx = str(uuid.uuid4())
        tmp_pth = f'/tmp/{idx}.png'
        cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), write_to=tmp_pth, dpi=200)
        image = cp.deepcopy(Image.open(tmp_pth))
        os.remove(tmp_pth)
        return image

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        contents = [self.SYSTEM_PROMPT_PREFIX]
        types = ['text']

        def _append_text(text):
            if types[-1] == 'text':
                contents[-1] += text
            else:
                contents.append(text)
                types.append('text')

        train_x = json.loads(line['train_x'])
        train_y = json.loads(line['train_y'])
        num_train = line['num_train']
        assert len(train_x) == len(train_y) == num_train
        for i in range(num_train):
            _append_text(f"--Example {i}-- \n\n INPUT: \n\n{json.dumps(train_x[i])}\n\n")
            if self.mode == 'image-text':
                _append_text("INPUT VISUALIZATION: \n\n")
                contents.append(self.generate_grid_image(train_x[i]))
                types.append('image')
            _append_text(f"OUTPUT: \n\n{json.dumps(train_y[i])}\n\n")
            if self.mode == 'image-text':
                _append_text("OUTPUT VISUALIZATION: \n\n")
                contents.append(self.generate_grid_image(train_y[i]))
                types.append('image')
        _append_text(self.SYSTEM_PROMPT_MID)
        test_x = json.loads(line['test_x'])
        _append_text(f'{test_x}\n\n')
        if self.mode == 'image-text':
            _append_text("TEST INPUT VISUALIZATION: \n\n")
            contents.append(self.generate_grid_image(test_x))
            types.append('image')
        _append_text(self.SYSTEM_PROMPT_SUFFIX)
        msgs = []
        for tp, content in zip(types, contents):
            if tp == 'text':
                msgs.append(dict(type='text', value=content))
            else:
                img_b64 = encode_image_to_base64(content, fmt='PNG')
                msgs.append(dict(type='image', value=f'data:image/png;base64,{img_b64}'))
        return msgs

    @staticmethod
    def extract_answer(response):
        arr = parse_json(response)
        if arr is None:
            return None
        if not isinstance(arr, list) or not all(isinstance(x, list) for x in arr):
            return None
        lens = [len(x) for x in arr]
        if not all(lens[0] == l for l in lens):
            return None
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                if arr[i][j] == '' or str(arr[i][j]) not in string.digits:
                    return None

        return arr

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file).sort_values(by='index')
        predictions = [str(x) for x in data['prediction']]
        pred_parsed = [self.extract_answer(x) for x in predictions]
        answers = [json.loads(row['test_y']) for _, row in data.iterrows()]
        judge_file = get_intermediate_file_path(eval_file, '_judge', 'tsv')
        rating_file = get_intermediate_file_path(eval_file, '_acc', 'csv')

        hit, mismatch = [], []
        for pred, ans in zip(pred_parsed, answers):
            if pred is None:
                hit.append(0)
                mismatch.append(-1)
                continue
            pred, ans = np.array(pred, dtype=np.int32), np.array(ans, dtype=np.int32)
            if pred.shape != ans.shape:
                hit.append(0)
                mismatch.append(-1)
                continue
            hit.append(int(np.all(pred == ans)))
            mismatch.append(np.sum(pred != ans))
        data['hit'] = hit
        data['mismatch'] = mismatch
        dump(data, judge_file)

        stat = defaultdict(list)
        for q, h in zip(data['task_index'], data['hit']):
            stat[q].append(h)

        strict = np.mean([np.all(val) for val in stat.values()])
        overall = np.mean([np.mean(val) for val in stat.values()])
        test_level = np.mean(data['hit'])
        format_err_rate = np.mean(data['mismatch'] == -1)
        res = {
            'overall': overall,
            'task_strict': strict,
            'test_average': test_level,
            'format_err_rate': format_err_rate
        }
        dump(d2df(res), rating_file)
        return res
