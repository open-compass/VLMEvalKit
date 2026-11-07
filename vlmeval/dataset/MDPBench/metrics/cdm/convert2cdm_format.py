import argparse
import json
import os

from tqdm import tqdm


def change_data_format(input_json, output_json):
    with open(input_json, 'r') as f:
        all_datas = json.load(f)

    data_list = []

    for key in all_datas.keys():
        subset = key[-4:-1].lower()
        for data in tqdm(all_datas[key]['text']):
            im_id = os.path.basename(data['image_path'])[0:-4]
            basename = f"{subset}_{im_id}"
            new_item = {
                "img_id": basename,
                "gt": data["reference"],
                "pred": data["prediction"]
            }
            data_list.append(new_item)

    with open(output_json, "w") as f:
        f.write(json.dumps(data_list, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--output', '-o', type=str)
    args = parser.parse_args()
    print(args)

    change_data_format(args.input, args.output)
