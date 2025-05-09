from collections import namedtuple
import importlib
import numpy as np
import Polygon as plg
from . import rrc_evaluation_funcs
import argparse
import json


def evaluation_imports():
    """
    evaluation_imports: Dictionary ( key = module name , value = alias  )  with python modules used in the evaluation.
    """
    return {
        'Polygon': 'plg',
        'numpy': 'np'
    }


def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
        'IOU_CONSTRAINT': 0.5,
        'AREA_PRECISION_CONSTRAINT': 0.5,
        'WORD_SPOTTING': False,
        'MIN_LENGTH_CARE_WORD': 3,
        'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID': 'res_img_([0-9]+).txt',
        'LTRB': False,  # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
        'CRLF': False,  # Lines are delimited by Windows CRLF format
        'CONFIDENCES': False,  # Detections must include confidence value. AP will be calculated
        'SPECIAL_CHARACTERS': '!?.:,*"()·[]/\'',
        'ONLY_REMOVE_FIRST_LAST_CHARACTER': True
    }


def validate_data(gtFilePath, submFilePath, evaluationParams):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
    Validates also that there are no missing files in the folder.
    If some error detected, the method raises the error
    """
    gt = rrc_evaluation_funcs.load_zip_file(
        gtFilePath, evaluationParams['GT_SAMPLE_NAME_2_ID']
    )

    subm = rrc_evaluation_funcs.load_zip_file(
        submFilePath, evaluationParams['DET_SAMPLE_NAME_2_ID'], True
    )

    # Validate format of GroundTruth
    for k in gt:
        rrc_evaluation_funcs.validate_lines_in_file(
            k, gt[k], evaluationParams['CRLF'], evaluationParams['LTRB'], True
        )

    # Validate format of results
    for k in subm:
        if k not in gt:
            raise Exception("The sample %s not present in GT" % k)

        rrc_evaluation_funcs.validate_lines_in_file(
            k, subm[k], evaluationParams['CRLF'], evaluationParams['LTRB'], True,
            evaluationParams['CONFIDENCES']
        )


def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
    Results. Dictionary with the following values:
    - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
    - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 }
    """
    for module, alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module)

    def polygon_from_points(points, correctOffset=False):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """
        if correctOffset:  # this will substract 1 from the coordinates that correspond to the xmax and ymax
            points[2] -= 1
            points[4] -= 1
            points[5] -= 1
            points[7] -= 1

        resBoxes = np.empty([1, 8], dtype='int32')
        resBoxes[0, 0] = int(points[0])
        resBoxes[0, 4] = int(points[1])
        resBoxes[0, 1] = int(points[2])
        resBoxes[0, 5] = int(points[3])
        resBoxes[0, 2] = int(points[4])
        resBoxes[0, 6] = int(points[5])
        resBoxes[0, 3] = int(points[6])
        resBoxes[0, 7] = int(points[7])
        pointMat = resBoxes[0].reshape([2, 4]).T
        return plg.Polygon(pointMat)

    def rectangle_to_polygon(rect):
        resBoxes = np.empty([1, 8], dtype='int32')
        resBoxes[0, 0] = int(rect.xmin)
        resBoxes[0, 4] = int(rect.ymax)
        resBoxes[0, 1] = int(rect.xmin)
        resBoxes[0, 5] = int(rect.ymin)
        resBoxes[0, 2] = int(rect.xmax)
        resBoxes[0, 6] = int(rect.ymin)
        resBoxes[0, 3] = int(rect.xmax)
        resBoxes[0, 7] = int(rect.ymax)

        pointMat = resBoxes[0].reshape([2, 4]).T
        return plg.Polygon(pointMat)

    def rectangle_to_points(rect):
        points = [
            int(rect.xmin), int(rect.ymax),
            int(rect.xmax), int(rect.ymax),
            int(rect.xmax), int(rect.ymin),
            int(rect.xmin), int(rect.ymin)
        ]
        return points

    def get_union(pD, pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - get_intersection(pD, pG)

    def get_intersection_over_union(pD, pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG)
        except:
            return 0

    def get_intersection(pD, pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()

    def compute_ap(confList, matchList, numGtCare):
        correct = 0
        AP = 0
        if len(confList) > 0:
            confList = np.array(confList)
            matchList = np.array(matchList)
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            for n in range(len(confList)):
                match = matchList[n]
                if match:
                    correct += 1
                    AP += float(correct) / (n + 1)

            if numGtCare > 0:
                AP /= numGtCare

        return AP

    def transcription_match(transGt, transDet, specialCharacters='!?.:,*"()·[]/\'', onlyRemoveFirstLastCharacterGT=True):  # noqa: E501
        if onlyRemoveFirstLastCharacterGT:
            # special characters in GT are allowed only at initial or final position
            if transGt == transDet:
                return True

            if specialCharacters.find(transGt[0]) > -1:
                if transGt[1:] == transDet:
                    return True

            if specialCharacters.find(transGt[-1]) > -1:
                if transGt[0:len(transGt) - 1] == transDet:
                    return True

            if (specialCharacters.find(transGt[0]) > -1 and  # noqa: W504
                    specialCharacters.find(transGt[-1]) > -1):
                if transGt[1:len(transGt) - 1] == transDet:
                    return True
            return False
        else:
            # Special characters are removed from the begining and the end of both Detection and GroundTruth
            while len(transGt) > 0 and specialCharacters.find(transGt[0]) > -1:
                transGt = transGt[1:]

            while len(transDet) > 0 and specialCharacters.find(transDet[0]) > -1:
                transDet = transDet[1:]

            while len(transGt) > 0 and specialCharacters.find(transGt[-1]) > -1:
                transGt = transGt[0:len(transGt) - 1]

            while len(transDet) > 0 and specialCharacters.find(transDet[-1]) > -1:
                transDet = transDet[0:len(transDet) - 1]

            return transGt == transDet

    def include_in_dictionary(transcription):
        # special case 's at final
        if transcription[len(transcription) - 2:] == "'s" or transcription[len(transcription) - 2:] == "'S":
            transcription = transcription[0:len(transcription) - 2]

        # hypens at init or final of the word
        transcription = transcription.strip('-')

        specialCharacters = "'!?.:,*\"()·[]/"
        for character in specialCharacters:
            transcription = transcription.replace(character, ' ')

        transcription = transcription.strip()

        if len(transcription) != len(transcription.replace(" ", "")):
            return False

        if len(transcription) < evaluationParams['MIN_LENGTH_CARE_WORD']:
            return False

        notAllowed = "×÷·"

        range1 = [ord(u'a'), ord(u'z')]
        range2 = [ord(u'A'), ord(u'Z')]
        range3 = [ord(u'À'), ord(u'ƿ')]
        range4 = [ord(u'Ǆ'), ord(u'ɿ')]
        range5 = [ord(u'Ά'), ord(u'Ͽ')]
        range6 = [ord(u'-'), ord(u'-')]

        for char in transcription:
            charCode = ord(char)
            if notAllowed.find(char) != -1:
                return False

            valid = (
                (charCode >= range1[0] and charCode <= range1[1]) or  # noqa: W504
                (charCode >= range2[0] and charCode <= range2[1]) or  # noqa: W504
                (charCode >= range3[0] and charCode <= range3[1]) or  # noqa: W504
                (charCode >= range4[0] and charCode <= range4[1]) or  # noqa: W504
                (charCode >= range5[0] and charCode <= range5[1]) or  # noqa: W504
                (charCode >= range6[0] and charCode <= range6[1])
            )
            if not valid:
                return False

        return True

    def include_in_dictionary_transcription(transcription):
        # special case 's at final
        if transcription[len(transcription) - 2:] == "'s" or transcription[len(transcription) - 2:] == "'S":
            transcription = transcription[0:len(transcription) - 2]

        # hypens at init or final of the word
        transcription = transcription.strip('-')

        specialCharacters = "'!?.:,*\"()·[]/"
        for character in specialCharacters:
            transcription = transcription.replace(character, ' ')

        transcription = transcription.strip()
        return transcription

    perSampleMetrics = {}

    matchedSum = 0

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath, evaluationParams['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath, evaluationParams['DET_SAMPLE_NAME_2_ID'], True)

    numGlobalCareGt = 0
    numGlobalCareDet = 0

    arrGlobalConfidences = []
    arrGlobalMatches = []

    for resFile in gt:
        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        if gtFile is None:
            raise Exception("The file %s is not UTF-8" % resFile)

        recall = 0
        precision = 0
        hmean = 0
        detCorrect = 0
        iouMat = np.empty([1, 1])
        gtPols = []
        detPols = []
        gtTrans = []
        detTrans = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCarePolsNum = []  # Array of Ground Truth Polygons' keys marked as don't Care
        detDontCarePolsNum = []  # Array of Detected Polygons' matched with a don't Care GT
        detMatchedNums = []
        pairs = []

        arrSampleConfidences = []
        arrSampleMatch = []
        sampleAP = 0

        evaluationLog = ""

        pointsList, _, transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(
            gtFile, evaluationParams['CRLF'], evaluationParams['LTRB'], True, False
        )
        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"
            if evaluationParams['LTRB']:
                gtRect = Rectangle(*points)
                gtPol = rectangle_to_polygon(gtRect)
            else:
                gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)

            # On word spotting we will filter some transcriptions with special characters
            if evaluationParams['WORD_SPOTTING']:
                if not dontCare:
                    if not include_in_dictionary(transcription):
                        dontCare = True
                    else:
                        transcription = include_in_dictionary_transcription(transcription)

            gtTrans.append(transcription)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += "GT polygons: " + str(len(gtPols)) + (
            " (" + str(len(gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum) > 0 else "\n"
        )

        if resFile in subm:
            detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile])

            pointsList, confidencesList, transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(  # noqa: E501
                detFile, evaluationParams['CRLF'], evaluationParams['LTRB'], True,
                evaluationParams['CONFIDENCES']
            )

            for n in range(len(pointsList)):
                points = pointsList[n]
                transcription = transcriptionsList[n]

                if evaluationParams['LTRB']:
                    detRect = Rectangle(*points)
                    detPol = rectangle_to_polygon(detRect)
                else:
                    detPol = polygon_from_points(points)
                detPols.append(detPol)
                detPolPoints.append(points)
                detTrans.append(transcription)

                if len(gtDontCarePolsNum) > 0:
                    for dontCarePol in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        intersected_area = get_intersection(dontCarePol, detPol)
                        pdDimensions = detPol.area()
                        precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if precision > evaluationParams['AREA_PRECISION_CONSTRAINT']:
                            detDontCarePolsNum.append(len(detPols) - 1)
                            break

            evaluationLog += "DET polygons: " + str(len(detPols)) + (
                " (" + str(len(detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum) > 0 else "\n"
            )

            if len(gtPols) > 0 and len(detPols) > 0:
                # Calculate IoU and precision matrixs
                outputShape = [len(gtPols), len(detPols)]
                iouMat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gtPols), np.int8)
                detRectMat = np.zeros(len(detPols), np.int8)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]
                        iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if (gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and  # noqa: W504
                                gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum):
                            if iouMat[gtNum, detNum] > evaluationParams['IOU_CONSTRAINT']:
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                # detection matched only if transcription is equal
                                if evaluationParams['WORD_SPOTTING']:
                                    correct = gtTrans[gtNum].upper() == detTrans[detNum].upper()
                                else:
                                    correct = transcription_match(
                                        gtTrans[gtNum].upper(), detTrans[detNum].upper(),
                                        evaluationParams['SPECIAL_CHARACTERS'],
                                        evaluationParams['ONLY_REMOVE_FIRST_LAST_CHARACTER']
                                    )
                                detCorrect += (1 if correct else 0)
                                if correct:
                                    detMatchedNums.append(detNum)
                                pairs.append({'gt': gtNum, 'det': detNum, 'correct': correct})
                                evaluationLog += (
                                    "Match GT #" + str(gtNum) + " with Det #" + str(detNum) +  # noqa: W504
                                    " trans. correct: " + str(correct) + "\n"
                                )

            if evaluationParams['CONFIDENCES']:
                for detNum in range(len(detPols)):
                    if detNum not in detDontCarePolsNum:
                        # we exclude the don't care detections
                        match = detNum in detMatchedNums

                        arrSampleConfidences.append(confidencesList[detNum])
                        arrSampleMatch.append(match)

                        arrGlobalConfidences.append(confidencesList[detNum])
                        arrGlobalMatches.append(match)

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
            sampleAP = precision
        else:
            recall = float(detCorrect) / numGtCare
            precision = 0 if numDetCare == 0 else float(detCorrect) / numDetCare
            if evaluationParams['CONFIDENCES']:
                sampleAP = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare)

        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        matchedSum += detCorrect
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics[resFile] = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'AP': sampleAP,
            'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
            'gtPolPoints': gtPolPoints,
            'detPolPoints': detPolPoints,
            'gtTrans': gtTrans,
            'detTrans': detTrans,
            'gtDontCare': gtDontCarePolsNum,
            'detDontCare': detDontCarePolsNum,
            'evaluationParams': evaluationParams,
            'evaluationLog': evaluationLog
        }

    # Compute AP
    AP = 0
    if evaluationParams['CONFIDENCES']:
        AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)  # noqa: E501

    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall, 'hmean': methodHmean, 'AP': AP}

    resDict = {'calculated': True, 'Message': '', 'method': methodMetrics, 'per_sample': perSampleMetrics}

    return resDict


def compute_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Compute detection metrics"""
    if not pred_boxes or not gt_boxes:
        return 0, 0, 0

    matched = [False] * len(gt_boxes)
    tp = 0

    for pred_box in pred_boxes:
        best_iou = 0
        best_idx = -1

        for i, gt_box in enumerate(gt_boxes):
            if matched[i]:
                continue
            iou = rrc_evaluation_funcs.compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_threshold:
            matched[best_idx] = True
            tp += 1

    precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0
    recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def evaluate_spotting(pred_results, gt_results, iou_threshold=0.5):
    """Evaluate spotting results"""
    # Initialize metrics
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count = 0

    for pred, gt in zip(pred_results, gt_results):
        if not pred or not gt:
            continue

        precision, recall, f1 = compute_metrics(pred, gt, iou_threshold)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        count += 1

    if count == 0:
        return 0, 0, 0

    avg_precision = total_precision / count
    avg_recall = total_recall / count
    avg_f1 = total_f1 / count

    return avg_precision, avg_recall, avg_f1


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate spotting results')
    parser.add_argument('--pred', type=str, required=True, help='Prediction file')
    parser.add_argument('--gt', type=str, required=True, help='Ground truth file')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
    args = parser.parse_args()

    # Load results
    with open(args.pred, 'r') as f:
        pred_results = json.load(f)
    with open(args.gt, 'r') as f:
        gt_results = json.load(f)

    # Evaluate
    precision, recall, f1 = evaluate_spotting(pred_results, gt_results, args.iou)

    # Print results
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')


if __name__ == '__main__':
    main()
