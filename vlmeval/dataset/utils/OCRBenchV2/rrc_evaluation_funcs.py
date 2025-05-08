import json
import sys
sys.path.append('./')
import zipfile
import re
import os
import numpy as np
import Polygon as plg
import argparse


def print_help():
    sys.stdout.write(
        'Usage: python %s.py -g=<gtFile> -s=<submFile> [-o=<outputFolder> -p=<jsonParams>]' % sys.argv[0]
    )
    sys.exit(2)


def load_zip_file_keys(file, fileNameRegExp=''):
    """
    Returns an array with the entries of the ZIP file that match with the regular expression.
    The key's are the names or the file or the capturing group definied in the fileNameRegExp
    """
    try:
        archive = zipfile.ZipFile(file, mode='r', allowZip64=True)
    except:
        raise Exception('Error loading the ZIP archive.')

    pairs = []

    for name in archive.namelist():
        addFile = True
        keyName = name
        if fileNameRegExp != "":
            m = re.match(fileNameRegExp, name)
            if m is None:
                addFile = False
            else:
                if len(m.groups()) > 0:
                    keyName = m.group(1)

        if addFile:
            pairs.append(keyName)

    return pairs


def load_zip_file(file, fileNameRegExp='', allEntries=False):
    """
    Returns an array with the contents (filtered by fileNameRegExp) of a ZIP file.
    The key's are the names or the file or the capturing group definied in the fileNameRegExp
    allEntries validates that all entries in the ZIP file pass the fileNameRegExp
    """
    try:
        archive = zipfile.ZipFile(file, mode='r', allowZip64=True)
    except:
        raise Exception('Error loading the ZIP archive')

    pairs = []
    for name in archive.namelist():
        addFile = True
        keyName = name
        if fileNameRegExp != "":
            m = re.match(fileNameRegExp, name)
            if m is None:
                addFile = False
            else:
                if len(m.groups()) > 0:
                    keyName = m.group(1)

        if addFile:
            pairs.append([keyName, archive.read(name)])
        else:
            if allEntries:
                raise Exception('ZIP entry not valid: %s' % name)

    return dict(pairs)


def decode_utf8(raw):
    """
    Returns a Unicode object on success, or None on failure
    """
    try:
        return raw.decode('utf-8-sig', errors='replace')
    except:
        return None


def validate_lines_in_file(fileName, file_contents, CRLF=True, LTRB=True, withTranscription=False, withConfidence=False, imWidth=0, imHeight=0):  # noqa: E501
    """
    This function validates that all lines of the file calling the Line validation function for each line
    """
    utf8File = decode_utf8(file_contents)
    if utf8File is None:
        raise Exception("The file %s is not UTF-8" % fileName)

    lines = utf8File.split("\r\n" if CRLF else "\n")
    for line in lines:
        line = line.replace("\r", "").replace("\n", "")
        if line != "":
            try:
                validate_tl_line(line, LTRB, withTranscription, withConfidence, imWidth, imHeight)
            except Exception as e:
                raise Exception(
                    ("Line in sample not valid. Sample: %s Line: %s Error: %s" % (
                        fileName, line, str(e)
                    )).encode('utf-8', 'replace')
                )


def validate_tl_line(line, LTRB=True, withTranscription=True, withConfidence=True, imWidth=0, imHeight=0):
    """
    Validate the format of the line. If the line is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values are:
    LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription]
    LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription]
    """
    get_tl_line_values(line, LTRB, withTranscription, withConfidence, imWidth, imHeight)


def get_tl_line_values(line, LTRB=True, withTranscription=False, withConfidence=False, imWidth=0, imHeight=0):
    """
    Validate the format of the line. If the line is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values are:
    LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription]
    LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription]
    Returns values from a textline. Points , [Confidences], [Transcriptions]
    """
    confidence = 0.0
    transcription = ""
    points = []

    numPoints = 4

    if LTRB:
        numPoints = 4

        if withTranscription and withConfidence:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',
                line
            )
            if m is None:
                m = re.match(
                    r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',
                    line
                )
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax,confidence,transcription")
        elif withConfidence:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$',
                line
            )
            if m is None:
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax,confidence")
        elif withTranscription:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,(.*)$',
                line
            )
            if m is None:
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax,transcription")
        else:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,?\s*$',
                line
            )
            if m is None:
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax")

        xmin = int(m.group(1))
        ymin = int(m.group(2))
        xmax = int(m.group(3))
        ymax = int(m.group(4))
        if xmax < xmin:
            raise Exception("Xmax value (%s) not valid (Xmax < Xmin)." % xmax)
        if ymax < ymin:
            raise Exception("Ymax value (%s)  not valid (Ymax < Ymin)." % ymax)

        points = [float(m.group(i)) for i in range(1, (numPoints + 1))]

        if imWidth > 0 and imHeight > 0:
            validate_point_inside_bounds(xmin, ymin, imWidth, imHeight)
            validate_point_inside_bounds(xmax, ymax, imWidth, imHeight)

    else:
        numPoints = 8

        if withTranscription and withConfidence:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',  # noqa: E501
                line
            )
            if m is None:
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence,transcription")
        elif withConfidence:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$',  # noqa: E501
                line
            )
            if m is None:
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence")
        elif withTranscription:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,(.*)$',  # noqa: E501
                line
            )
            if m is None:
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,transcription")
        else:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*$',  # noqa: E501
                line
            )
            if m is None:
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4")

        points = [float(m.group(i)) for i in range(1, (numPoints + 1))]

        validate_clockwise_points(points)

        if imWidth > 0 and imHeight > 0:
            validate_point_inside_bounds(points[0], points[1], imWidth, imHeight)
            validate_point_inside_bounds(points[2], points[3], imWidth, imHeight)
            validate_point_inside_bounds(points[4], points[5], imWidth, imHeight)
            validate_point_inside_bounds(points[6], points[7], imWidth, imHeight)

    if withConfidence:
        try:
            confidence = float(m.group(numPoints + 1))
        except ValueError:
            raise Exception("Confidence value must be a float")

    if withTranscription:
        posTranscription = numPoints + (2 if withConfidence else 1)
        transcription = m.group(posTranscription)
        m2 = re.match(r'^\s*\"(.*)\"\s*$', transcription)
        if m2 is not None:  # Transcription with double quotes, we extract the value and replace escaped characters
            transcription = m2.group(1).replace("\\\\", "\\").replace("\\\"", "\"")

    return points, confidence, transcription


def get_tl_dict_values(detection, withTranscription=False, withConfidence=False, imWidth=0, imHeight=0,
                       validNumPoints=[], validate_cw=True):
    """
    Validate the format of the dictionary. If the dictionary is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values:
    {"points":[[x1,y1],[x2,y2],[x3,x3],..,[xn,yn]]}
    {"points":[[x1,y1],[x2,y2],[x3,x3],..,[xn,yn]],"transcription":"###","confidence":0.4,"illegibility":false}
    {"points":[[x1,y1],[x2,y2],[x3,x3],..,[xn,yn]],"transcription":"###","confidence":0.4,"dontCare":false}
    Returns values from the dictionary. Points , [Confidences], [Transcriptions]
    """
    confidence = 0.0
    transcription = ""
    points = []

    if not isinstance(detection, dict):
        raise Exception("Incorrect format. Object has to be a dictionary")

    if 'points' not in detection:
        raise Exception("Incorrect format. Object has no points key)")

    if not isinstance(detection['points'], list):
        raise Exception("Incorrect format. Object points key have to be an array)")

    num_points = len(detection['points'])

    if num_points < 3:
        raise Exception(
            "Incorrect format. Incorrect number of points. At least 3 points are necessary. Found: " + str(num_points)
        )

    if len(validNumPoints) > 0 and num_points not in validNumPoints:
        raise Exception("Incorrect format. Incorrect number of points. Only allowed 4,8 or 12 points)")

    for i in range(num_points):
        if not isinstance(detection['points'][i], list):
            raise Exception("Incorrect format. Point #" + str(i + 1) + " has to be an array)")

        if len(detection['points'][i]) != 2:
            raise Exception("Incorrect format. Point #" + str(i + 1) + " has to be an array with 2 objects(x,y) )")

        if not isinstance(detection['points'][i][0], (int, float)) or not isinstance(detection['points'][i][1], (int, float)):  # noqa: E501
            raise Exception("Incorrect format. Point #" + str(i + 1) + " childs have to be Integers)")

        if imWidth > 0 and imHeight > 0:
            validate_point_inside_bounds(detection['points'][i][0], detection['points'][i][1], imWidth, imHeight)

        points.append(float(detection['points'][i][0]))
        points.append(float(detection['points'][i][1]))

    if validate_cw:
        validate_clockwise_points(points)

    if withConfidence:
        if 'confidence' not in detection:
            raise Exception("Incorrect format. No confidence key)")

        if not isinstance(detection['confidence'], (int, float)):
            raise Exception("Incorrect format. Confidence key has to be a float)")

        if detection['confidence'] < 0 or detection['confidence'] > 1:
            raise Exception("Incorrect format. Confidence key has to be a float between 0.0 and 1.0")

        confidence = detection['confidence']

    if withTranscription:
        if 'transcription' not in detection:
            raise Exception("Incorrect format. No transcription key)")

        if not isinstance(detection['transcription'], str):
            raise Exception(
                "Incorrect format. Transcription has to be a string. Detected: " +  # noqa: W504
                type(detection['transcription']).__name__
            )

        transcription = detection['transcription']

        if 'illegibility' in detection:  # Ensures that if illegibility
            if detection['illegibility']:
                transcription = "###"

        if 'dontCare' in detection:  # Ensures that if dontCare
            if detection['dontCare']:
                transcription = "###"

    return points, confidence, transcription


def validate_point_inside_bounds(x, y, imWidth, imHeight):
    if x < 0 or x > imWidth:
        raise Exception("X value not valid. Image dimensions: (%s,%s)" % (imWidth, imHeight))
    if y < 0 or y > imHeight:
        raise Exception("Y value not valid. Image dimensions: (%s,%s)" % (imWidth, imHeight))


def validate_clockwise_points(points):
    """
    Validates that the points are in clockwise order.
    """
    edge = []
    for i in range(len(points) // 2):
        edge.append(
            (int(points[(i + 1) * 2 % len(points)]) - int(points[i * 2])) *  # noqa: W504
            (int(points[((i + 1) * 2 + 1) % len(points)]) + int(points[i * 2 + 1]))
        )
    if sum(edge) > 0:
        raise Exception(
            "Points are not clockwise. The coordinates of bounding points have to be given in clockwise order. "
            "Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is "
            "the standard one, with the image origin at the upper left, the X axis extending to the right and "
            "Y axis extending downwards."
        )


def get_tl_line_values_from_file_contents(content, CRLF=True, LTRB=True, withTranscription=False, withConfidence=False, imWidth=0, imHeight=0, sort_by_confidences=True):  # noqa: 501
    """
    Returns all points, confindences and transcriptions of a file in lists. Valid line formats:
    xmin,ymin,xmax,ymax,[confidence],[transcription]
    x1,y1,x2,y2,x3,y3,x4,y4,[confidence],[transcription]
    """
    pointsList = []
    transcriptionsList = []
    confidencesList = []

    lines = content.split("\r\n" if CRLF else "\n")
    for line in lines:
        line = line.replace("\r", "").replace("\n", "")
        if line != "":
            points, confidence, transcription = get_tl_line_values(
                line, LTRB, withTranscription, withConfidence, imWidth, imHeight
            )
            pointsList.append(points)
            transcriptionsList.append(transcription)
            confidencesList.append(confidence)

    if withConfidence and len(confidencesList) > 0 and sort_by_confidences:
        sorted_ind = np.argsort(-np.array(confidencesList))
        confidencesList = [confidencesList[i] for i in sorted_ind]
        pointsList = [pointsList[i] for i in sorted_ind]
        transcriptionsList = [transcriptionsList[i] for i in sorted_ind]

    return pointsList, confidencesList, transcriptionsList


def get_tl_dict_values_from_array(array, withTranscription=False, withConfidence=False, imWidth=0, imHeight=0, sort_by_confidences=True, validNumPoints=[], validate_cw=True):  # noqa: 501
    """
    Returns all points, confindences and transcriptions of a file in lists. Valid dict formats:
    {"points":[[x1,y1],[x2,y2],[x3,x3],..,[xn,yn]],"transcription":"###","confidence":0.4}
    """
    pointsList = []
    transcriptionsList = []
    confidencesList = []

    for n in range(len(array)):
        objectDict = array[n]
        points, confidence, transcription = get_tl_dict_values(
            objectDict, withTranscription, withConfidence, imWidth, imHeight, validNumPoints, validate_cw
        )
        pointsList.append(points)
        transcriptionsList.append(transcription)
        confidencesList.append(confidence)

    if withConfidence and len(confidencesList) > 0 and sort_by_confidences:
        sorted_ind = np.argsort(-np.array(confidencesList))
        confidencesList = [confidencesList[i] for i in sorted_ind]
        pointsList = [pointsList[i] for i in sorted_ind]
        transcriptionsList = [transcriptionsList[i] for i in sorted_ind]

    return pointsList, confidencesList, transcriptionsList


def main_evaluation(p, default_evaluation_params, validate_data, evaluate_method):
    """Main evaluation function."""
    if p is not None:
        parser = argparse.ArgumentParser(description='Evaluate detection results')
        parser.add_argument('--gt', type=str, required=True, help='Ground truth file')
        parser.add_argument('--det', type=str, required=True, help='Detection file')
        parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
        args = parser.parse_args()

        p = {
            'g': args.gt,
            's': args.det,
            'o': args.iou
        }

    if p is None:
        parser = argparse.ArgumentParser(description='Evaluate detection results')
        parser.add_argument('--gt', type=str, required=True, help='Ground truth file')
        parser.add_argument('--det', type=str, required=True, help='Detection file')
        parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
        args = parser.parse_args()

        p = {
            'g': args.gt,
            's': args.det,
            'o': args.iou
        }

    eval_params = default_evaluation_params()
    if 'o' in p:
        eval_params['IOU_CONSTRAINT'] = float(p['o'])

    validate_data(p['g'], p['s'], eval_params)
    res = evaluate_method(p['g'], p['s'], eval_params)

    print('Precision: %.4f' % res['method']['precision'])
    print('Recall: %.4f' % res['method']['recall'])
    print('F1: %.4f' % res['method']['hmean'])
    if 'AP' in res['method']:
        print('AP: %.4f' % res['method']['AP'])

    return res


def main_validation(default_evaluation_params_fn, validate_data_fn):
    try:
        p = dict([s[1:].split('=') for s in sys.argv[1:]])
        evalParams = default_evaluation_params_fn()
        if 'p' in p.keys():
            evalParams.update(p['p'] if isinstance(p['p'], dict) else json.loads(p['p']))

        validate_data_fn(p['g'], p['s'], evalParams)
        print('SUCCESS')
        sys.exit(0)
    except Exception as e:
        print(str(e))
        sys.exit(101)


def compute_intersection(box1, box2):
    """Compute intersection between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 > x2 or y1 > y2:
        return 0
    return (x2 - x1) * (y2 - y1)


def compute_area(box):
    """Compute area of a box"""
    return (box[2] - box[0]) * (box[3] - box[1])


def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    intersection = compute_intersection(box1, box2)
    area1 = compute_area(box1)
    area2 = compute_area(box2)
    union = area1 + area2 - intersection
    if union == 0:
        return 0
    return intersection / union


def compute_precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Compute precision and recall"""
    if pred_boxes is None or gt_boxes is None:
        return 0, 0

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0, 0

    matched = [False] * len(gt_boxes)
    tp = 0

    for pred_box in pred_boxes:
        best_iou = 0
        best_idx = -1

        for i, gt_box in enumerate(gt_boxes):
            if matched[i]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_threshold:
            matched[best_idx] = True
            tp += 1

    precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0
    recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0

    return precision, recall


def compute_f1_score(precision, recall):
    """Compute F1 score"""
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def evaluate_detection(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Evaluate detection results"""
    precision, recall = compute_precision_recall(pred_boxes, gt_boxes, iou_threshold)
    f1 = compute_f1_score(precision, recall)
    return precision, recall, f1


def evaluate_recognition(pred_texts, gt_texts):
    """Evaluate recognition results"""
    if pred_texts is None or gt_texts is None:
        return 0, 0, 0

    if len(pred_texts) == 0 or len(gt_texts) == 0:
        return 0, 0, 0

    correct = 0
    for pred, gt in zip(pred_texts, gt_texts):
        if pred == gt:
            correct += 1

    precision = correct / len(pred_texts) if len(pred_texts) > 0 else 0
    recall = correct / len(gt_texts) if len(gt_texts) > 0 else 0
    f1 = compute_f1_score(precision, recall)

    return precision, recall, f1


def evaluate_end_to_end(pred_boxes, pred_texts, gt_boxes, gt_texts, iou_threshold=0.5):
    """Evaluate end-to-end results"""
    if pred_boxes is None or gt_boxes is None or pred_texts is None or gt_texts is None:
        return 0, 0, 0

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0, 0, 0

    matched = [False] * len(gt_boxes)
    tp = 0

    for pred_box, pred_text in zip(pred_boxes, pred_texts):
        best_iou = 0
        best_idx = -1

        for i, (gt_box, gt_text) in enumerate(zip(gt_boxes, gt_texts)):
            if matched[i]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_threshold and pred_text == gt_texts[best_idx]:
            matched[best_idx] = True
            tp += 1

    precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0
    recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
    f1 = compute_f1_score(precision, recall)

    return precision, recall, f1
