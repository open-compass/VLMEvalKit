import json
import os
import copy
import pandas as pd
from tqdm import tqdm
import Levenshtein
import tempfile
from collections import deque
import base64
from collections import defaultdict
import numpy as np
from lxml import etree, html
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html

from utils import match_gt2pred_simple, match_gt2pred_no_split,match_gt2pred_quick,md_tex_filter
from metrics import show_result, get_full_labels_results, get_page_split
from metrics import METRIC_REGISTRY,recogition_end2end_base_dataset,recogition_end2end_table_dataset

from func_timeout import FunctionTimedOut, func_timeout



eval_file='/mnt/petrelfs/wangfangdong/wang/VLMEvalKit/outputs/Qwen2-VL-2B-Instruct/Qwen2-VL-2B-Instruct_OmniDocBench.xlsx'
tsv_path='/mnt/petrelfs/wangfangdong/LMUData/OmniDocBench.tsv'

class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.



class call_TEDS():
    def __init__(self, samples):
        self.samples = samples
        # type(samples):<class 'metrics.recogition_end2end_table_dataset'>
        print(f'type(samples):{type(samples)}') #dict {{},{},{}}
        print(samples)
     
        
        """
        {'gt_idx': [0], 
        'gt': '<table>\n<thead>\n<tr>\n <th>得分</th>\n <th></th>\n</tr>\n</thead>\n<tbody>\n<tr>\n <td>阅卷人</td>\n <td></td>\n</tr>\n</tbody>\n</table>',
        'pred_idx': [''], 'pred': '', 'gt_position': [10], 
        'pred_position': '', 
        'norm_gt': '<html><body><table border="1" ><tr><td>得分</td><td></td></tr><tr><td>阅卷人</td><td></td></tr></table></body></html>', 
        'norm_pred': '', 
        'gt_category_type': 'table',
        'pred_category_type': '',
        'gt_attribute': [{'table_layout': 'horizontal',
        'with_span': False, 'line': 'full_line', 'language': 'table_simplified_chinese', 'include_equation': False, 'include_photo': False, 'include_background': False, 'with_structured_text': False}], 'edit': 1, 
        'img_id': 'jiaocaineedrop_38247658.pdf_0.jpg'}
        """
    def evaluate(self, group_info=[], save_name='default'):

        teds = TEDS(structure_only=False)
        teds_structure_only = TEDS(structure_only=True)

        group_scores = defaultdict(list)
        group_scores_structure_only = defaultdict(list)
        samples = self.samples

        for i,sample in enumerate(samples):

            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
          
            score = teds.evaluate(pred, gt)
            score_structure_only = teds_structure_only.evaluate(pred, gt)
            
            if i==79 or i==145 or i==376 or i==377:
                print(f'第{i}个元素有得分')
                print('TEDS score:', score)
                # print(sample)
                # print(f'pred:{pred}')
                # print(f'gt:{gt}')

                

            group_scores['all'].append(score)
            group_scores_structure_only['all'].append(score_structure_only)
            if not sample.get('metric'):
                sample['metric'] = {}
            sample['metric']['TEDS'] = score
            sample['metric']['TEDS_structure_only'] = score_structure_only
            for group in group_info:
                select_flag = True
                for k, v in group.items():
                    for gt_attribute in sample['gt_attribute']:   # gt_attribute is a list containing all merged gt attributes
                        if not gt_attribute:   # if no GT attributes, don't include in calculation
                            select_flag = False
                        elif gt_attribute[k] != v:  # if any gt attribute doesn't meet criteria, don't select
                            select_flag = False
                if select_flag:
                    group_scores[str(group)].append(score)
         
        

        result = {}
        for group_name, scores in group_scores.items():
            if len(scores) > 0:
                result[group_name] = sum(scores) / len(scores)    # average of normalized scores at sample level
            else:
                result[group_name] = 'NaN'
                print(f'Warning: Empyty matched samples for {group_name}.')
        
        structure_only_result = {}
        for group_name, scores in group_scores_structure_only.items():
            if len(scores) > 0:
                structure_only_result[group_name] = sum(scores) / len(scores)    # average of normalized scores at sample level
            else:
                structure_only_result[group_name] = 'NaN'
                print(f'Warning: Empyty matched samples for {group_name}.')

        return samples, {'TEDS': result, 'TEDS_structure_only': structure_only_result}


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0

    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        samples = true_json.keys()
        # if self.n_jobs == 1:
        scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        # else:
        #     inputs = [{'pred': pred_json.get(filename, ''), 'true': true_json[filename]['html']} for filename in samples]
        #     scores = parallel_process(inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        scores = dict(zip(samples, scores))
        return scores



class Omnidocbenchend2endEvaluator():
    def __init__(self,
                 eval_file,
                 tsv_path,
                 match_method:str='quick_match',
                 filter_types:dict=None):
        
        self.eval_file=eval_file
        self.match_method=match_method
        self.references=[]
        self.predictions = pd.read_excel(eval_file)['prediction'].tolist()
        self.dafault_metircs_dict={
            'text_block': 
                {'metric': ['Edit_dist', 'BLEU', 'METEOR']},
            'display_formula': 
                {'metric': ['Edit_dist', 'CDM']}, 
            'table':
                {'metric': ['TEDS', 'Edit_dist']}, 
            'reading_order': 
                {'metric': ['Edit_dist']}
            }
        
        references = pd.read_csv(tsv_path, sep='\t')['answer'].tolist()

        load_success,load_fail=0,0
        # str->dict
        for i,ans in enumerate(references):
            try:
                ans = json.loads(ans)
                load_success+=1
                self.references.append(ans) #[{},{}]
            except json.JSONDecodeError as e:
                load_fail+=1
                continue
        print(f'load_success:{load_success},load_fail:{load_fail}')
           
        filtered_gt_samples = []
        if filter_types:
            for gt_sample in self.references:
                select_flag = True
                for k, v in filter_types.items():
                    if gt_sample["page_info"]["page_attribute"][k] != v:
                        select_flag = False
                if select_flag:
                    filtered_gt_samples.append(gt_sample)
        else:
            filtered_gt_samples = self.references #[{},{},{}]
        self.references=filtered_gt_samples
      

    def score(self)->dict:
        samples=self.get_matched_elements(self.references,self.predictions)
        metrics=self.process_generated_metric_results(samples)
        return metrics      

    def get_page_elements(self, selected_annos):
        saved_element_dict = defaultdict(list) 
        related_truncated = [] 
        truncated_all = {}
        for relation in selected_annos["extra"]["relation"]:   # Handle truncated text issues
            if relation["relation_type"] == 'truncated':
                truncated_all[relation["source_anno_id"]] = ""
                truncated_all[relation["target_anno_id"]] = ""
                exist_flag = False
                for merge_list in related_truncated:
                    if relation["source_anno_id"] in merge_list or relation["target_anno_id"] in merge_list:  # Consider cases where three text blocks may need to be merged
                        merge_list.append(relation["source_anno_id"])
                        merge_list.append(relation["target_anno_id"])
                        exist_flag = True
                if not exist_flag:
                    related_truncated.append([relation["source_anno_id"], relation["target_anno_id"]])       
        
        for item in selected_annos['layout_dets']:
            if item['anno_id'] not in truncated_all.keys():
                saved_element_dict[item["category_type"]].append(item)
            else:
                truncated_all[item['anno_id']] = item
        
        for merge_list in related_truncated:
            text_block_list = [truncated_all[key] for key in merge_list]
            sorted_block = sorted(text_block_list, key=lambda x: x['order'])
            text = ""
            for block in sorted_block:
                text += block['text']
            merged_block = {
                "category_type": sorted_block[0]["category_type"], # Directly use information from the first block
                "order": sorted_block[0]["order"],
                "anno_id": sorted_block[0]["anno_id"],   
                "text": text,
                "merge_list": sorted_block
            }
            saved_element_dict[sorted_block[0]["category_type"]].append(merged_block)
            
        return saved_element_dict
    
    def get_page_elements_list(self, gt_page_elements, category_list):
        element_list = []
        for category_type in category_list:
            if gt_page_elements.get(category_type):
                element_list.extend(gt_page_elements[category_type])
        return element_list
    
    def get_sorted_text_list(self, selected_annos):
        # txt_type: text, latex, html
        text_list = []
        for item in selected_annos:
            if item.get('order'):
                order = item['order']
            else:
                order = 0
            # 【txt_type,selecte_annos]
            text_list.append((order, item))
        sorted_text_list = sorted(text_list, key=lambda x: x[0])
        return [_[1] for _ in sorted_text_list]
    
    def filtered_out_ignore(self, items, ignore_category_list):
        filted_items = []
        for item in items:
            if item['gt_category_type'] not in ignore_category_list:
                filted_items.append(item)
        return filted_items

    def get_order_paired(self, order_match_s, img_name):
        matched = [(item['gt_position'], item['pred_position']) for item in order_match_s if (item['gt_position'] != [""] and item['pred_position'] != "")]
        gt_idx_all = [item['gt_position'] for item in order_match_s if (item['gt_position'] != [""])]
        read_order_pred = [i[0] for i in sorted(matched, key=lambda x: x[1])]  
        read_order_gt = sum(gt_idx_all, []) # Convert to one-dimensional list
        read_order_gt = [x for x in read_order_gt if x]  
        gt = sorted(read_order_gt) 
        pred = sum(read_order_pred, [])
        pred = [x for x in pred if x]
        if len(pred) > 0 or len(gt) > 0:
            edit = Levenshtein.distance(gt, pred)/ max(len(pred), len(gt))
            return {
                'gt': gt,  
                'pred': pred,
                'img_id': img_name,
                'edit': edit
            }
        else:
            return {}  # If both GT and pred are empty for the page, return empty

    def formula_format(self, formula_matches, img_name):
        # formated_list = []
        for i, item in enumerate(formula_matches):
            item["img_id"] = img_name + '_' + str(i)
        return formula_matches
            
    def get_matched_elements(self,references:list,predictions:list)->dict:

        plain_text_match = []
        display_formula_match = []
        html_table_match = []
        latex_table_match = []
        order_match = []
  

        for i,sample in enumerate(references):
            img_name = os.path.basename(sample["page_info"]["image_path"])
            pred_content = predictions[i]
            result = self.process_get_matched_elements(sample, pred_content, img_name) 
            [plain_text_match_clean, formated_display_formula, latex_table_match_s, html_table_match_s, order_match_single] = result

            if order_match_single:
                order_match.append(order_match_single)
            if plain_text_match_clean:
                plain_text_match.extend(plain_text_match_clean)
            if formated_display_formula:
                display_formula_match.extend(formated_display_formula)
            if latex_table_match_s:
                latex_table_match.extend(latex_table_match_s)
            if html_table_match_s:
                html_table_match.extend(html_table_match_s)

        if len(latex_table_match) > len(html_table_match): 
            table_match = latex_table_match
            table_format = 'latex'
        else:
            table_match = html_table_match
            table_format = 'html'

        matched_samples_all = {
            'text_block': recogition_end2end_base_dataset(plain_text_match),
            'display_formula': recogition_end2end_base_dataset(display_formula_match),
            'table': recogition_end2end_table_dataset(table_match, table_format),
            'reading_order': recogition_end2end_base_dataset(order_match)
        }
        
        return matched_samples_all
    
    def process_get_matched_elements(self, sample, pred_content, img_name):
        if self.match_method == 'simple_match':   # add match choice
            match_gt2pred = match_gt2pred_simple
        elif self.match_method == 'quick_match':
            match_gt2pred = match_gt2pred_quick
        elif self.match_method == 'no_split':
            match_gt2pred = match_gt2pred_no_split
        else:
            # print('Invalid match method name. The quick_match will be used.')
            match_gt2pred = match_gt2pred_quick

        pred_dataset = md_tex_filter(pred_content)
        gt_page_elements = self.get_page_elements(sample)
        
        text_all = self.get_page_elements_list(gt_page_elements, ['text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption',
                                                'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption',
                                                'header', 'footer', 'page_footnote', 'page_number'])

        
        display_formula_match_s = []
        plain_text_match_clean = []
        latex_table_match_s = []
        html_table_match_s = []
        order_match_single = []
        if text_all:
            gt_text_list = self.get_sorted_text_list(text_all)
            try:
                plain_text_match_s = func_timeout(
                    30, match_gt2pred, args=(gt_text_list, pred_dataset['text_all'], 'text', img_name)
                )
            except FunctionTimedOut as e1:
                print(f'Time out for plain text match of {img_name}, match_gt2pred_simple will be used.')
                plain_text_match_s = match_gt2pred_simple(gt_text_list, pred_dataset['text_all'], 'text', img_name)
            except Exception as e:
                print(str(e))
                sys.exit()     

            if not plain_text_match_s:
                print(f'No text match of {img_name}. The plain text match will be empty.')
            else:
                plain_text_match_clean = self.filtered_out_ignore(plain_text_match_s, ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption'])
            

        if gt_page_elements.get('equation_isolated'):
            gt_display_list = self.get_sorted_text_list(gt_page_elements['equation_isolated'])
            display_formula_match_s = match_gt2pred(gt_display_list, pred_dataset['equation_isolated'], 'formula', img_name)
            display_formula_match_s = [x for x in display_formula_match_s if x['gt_idx'] != [""]]  
            if not display_formula_match_s:
                print(f'No display_formula_match of {img_name}. The display_formula_match will be empty.')
      
        if gt_page_elements.get('table'):
            gt_table_list = self.get_sorted_text_list(gt_page_elements['table'])
            if pred_dataset['latex_table']:
                latex_table_match_s = match_gt2pred_simple(gt_table_list, pred_dataset['latex_table'], 'latex_table', img_name) 
                latex_table_match_s = [x for x in latex_table_match_s if x['gt_idx'] != [""]] 
            if pred_dataset['html_table']:  
                html_table_match_s = match_gt2pred_simple(gt_table_list, pred_dataset['html_table'], 'html_table', img_name) 
                html_table_match_s = [x for x in html_table_match_s if x['gt_idx'] != [""]]  
            else:
                html_table_match_s = match_gt2pred_simple(gt_table_list, [], 'html_table', img_name) 
                html_table_match_s = [x for x in html_table_match_s if x['gt_idx'] != [""]]  


        order_match_s = plain_text_match_clean
        if order_match_s:
            order_match_single = self.get_order_paired(order_match_s, img_name)
            
        return [plain_text_match_clean, display_formula_match_s, latex_table_match_s, html_table_match_s, order_match_single]  


    def process_generated_metric_results(self,samples,save_name:str='end2end_quick_match'):
        
        result_all={}
        page_info={}
        metircs_dict=self.dafault_metircs_dict
        pages=self.references #gt_samples list
       
        for page in pages:
                img_path=os.path.basename(page['page_info']['image_path'])
                page_info[img_path]=page['page_info']['page_attribute']

        for element in metircs_dict.keys():
            result={}
            group_info=metircs_dict[element].get('group',[])
            sample = samples.get(element)

            for metric in metircs_dict[element]['metric']:
                metric_val = METRIC_REGISTRY.get(metric)

                sample,result_s = metric_val(sample).evaluate(group_info, f"{save_name}_{element}")
                if result_s:
                    result.update(result_s)
                # if isinstance(result_s, tuple) and len(result_s) > 1 and isinstance(result_s[1], dict):
                #     result.update(result_s[1])    
            if result:
                print(f"{element}")
                show_result(result)
            result_all[element]={}

        
            group_result=get_full_labels_results(sample)
            page_result=get_page_split(sample,page_info)

            result_all[element]={
                'all':result,
                'group':group_result,
                'page':page_result
            }

        dict_list = []
        save_dict={}
        en_overall=[]
        ch_overall=[]
        for category_type, metric in [("text_block", "Edit_dist"), ("display_formula", "Edit_dist"), ("display_formula", "CDM"), ("table", "TEDS"), ("table", "Edit_dist"), ("reading_order", "Edit_dist")]:
            if metric == 'CDM':
                save_dict[category_type+'_'+metric+'_EN'] = '-'
                save_dict[category_type+'_'+metric+'_CH'] = '-'
            elif metric == "TEDS":
                save_dict[category_type+'_'+metric+'_EN'] = result_all[category_type]["page"][metric]["language: english"] * 100
                save_dict[category_type+'_'+metric+'_CH'] = result_all[category_type]["page"][metric]["language: simplified_chinese"] * 100
            else:
                save_dict[category_type+'_'+metric+'_EN'] = result_all[category_type]["page"][metric].get("language: english", np.nan)
                save_dict[category_type+'_'+metric+'_CH'] = result_all[category_type]["page"][metric].get("language: simplified_chinese",np.nan)
            if metric == "Edit_dist":
                en_overall.append(result_all[category_type]["page"][metric].get("language: english", np.nan))
                ch_overall.append(result_all[category_type]["page"][metric].get("language: simplified_chinese",np.nan))
        
        save_dict['overall_EN'] = sum(en_overall) / len(en_overall)
        save_dict['overall_CH'] = sum(ch_overall) / len(ch_overall)
        dict_list.append(save_dict)
        df = pd.DataFrame(dict_list,index=['end2end',]).round(3)

        return df



eval=Omnidocbenchend2endEvaluator(eval_file,tsv_path)
result=eval.score()
print(result)




















# print(result['overall_CH'])

#sample
"""
{'gt_idx': [0], 
'gt': '<table>\n<thead>\n<tr>\n <th colspan="4" rowspan="2">九毛九国际控股有限公司(09922)</th>\n</tr>\n<tr>\n</tr>\n</thead>\n<tbody>\n<tr>\n <td>总市值</td>\n <td>营收规模</td>\n <td>同比增长(%)</td>\n <td>毛利率(%)</td>\n</tr>\n<tr>\n <td>88.0亿</td>\n <td>41.8亿</td>\n <td>53.9600</td>\n <td>_ _</td>\n</tr>\n</tbody>\n</table>', 
'pred_idx': [''], 
'pred': '', 
'gt_position': [4], 
'pred_position': '',
'norm_gt': '<html><body><table border="1" ><tr><td colspan="4" rowspan="2">九毛九国际控股有限公司(09922)</td></tr><tr></tr><tr><td>总市值</td><td>营收规模</td><td>同比增长(%)</td><td>毛利率(%)</td></tr><tr><td>88.0亿</td><td>41.8亿</td><td>53.9600</td><td>_ _</td></tr></table></body></html>', 
'norm_pred': '', 
'gt_category_type': 'table',
'pred_category_type': '', 
'gt_attribute': [{'table_layout': 'horizontal', 'with_span': True, 'line': 'wireless_line', 'language': 'table_simplified_chinese', 'include_equation': False, 'include_photo': False, 'include_background': True, 'with_structured_text': False}], 'edit': 1,
'img_id': 'yanbaopptmerge_70a45bf10fb90ae0c91d44a11e0a97a8f9515d85626b54eb50a80b6146e5c2bf.pdf_10.jpg', 
'metric': {'TEDS': 0.0, 'TEDS_structure_only': 0.0}}


{'gt_idx': [0],
'gt': '<table>\n<thead>\n<tr>\n <th colspan="4" rowspan="2">九毛九国际控股有限公司(09922)</th>\n</tr>\n<tr>\n</tr>\n</thead>\n<tbody>\n<tr>\n <td>总市值</td>\n <td>营收规模</td>\n <td>同比增长(%)</td>\n <td>毛利率(%)</td>\n</tr>\n<tr>\n <td>88.0亿</td>\n <td>41.8亿</td>\n <td>53.9600</td>\n <td>_ _</td>\n</tr>\n</tbody>\n</table>',
'norm_gt': '<html><body><table border="1" ><tr><td colspan="4" rowspan="2">九毛九国际控股有限公司(09922)</td></tr><tr></tr><tr><td>总市值</td><td>营收规模</td><td>同比增长(%)</td><td>毛利率(%)</td></tr><tr><td>88.0亿</td><td>41.8亿</td><td>53.9600</td><td>_ _</td></tr></table></body></html>', 
'gt_category_type': 'table',
'gt_position': [4], 
'gt_attribute': [{'table_layout': 'horizontal', 'with_span': True, 'line': 'wireless_line', 'language': 'table_simplified_chinese', 'include_equation': False, 'include_photo': False, 'include_background': True, 'with_structured_text': False}], 
'pred_idx': [0],
'pred': '<table border="1" >\n    <tr>\n      <td colspan="1" rowspan="1">总市值</td>\n      <td colspan="1" rowspan="1">营收规模</td>\n      <td colspan="1" rowspan="1">同比增长(%)</td>\n      <td colspan="1" rowspan="1">毛利率(%)</td>\n    </tr>\n    <tr>\n      <td colspan="1" rowspan="1">88.0亿</td>\n      <td colspan="1" rowspan="1">41.8亿</td>\n      <td colspan="1" rowspan="1">53.9600</td>\n      <td colspan="1" rowspan="1">--</td>\n    </tr>\n</table>', 
'norm_pred': '<html><body><table border="1" ><tr><td colspan="1" rowspan="1">总市值</td><td colspan="1" rowspan="1">营收规模</td><td colspan="1" rowspan="1">同比增长(%)</td><td colspan="1" rowspan="1">毛利率(%)</td></tr><tr><td colspan="1" rowspan="1">88.0亿</td><td colspan="1" rowspan="1">41.8亿</td><td colspan="1" rowspan="1">53.9600</td><td colspan="1" rowspan="1">--</td></tr></table></body></html>', 
'pred_category_type': '', 
'pred_position': '',
'edit': 0.6423529411764706, 
'img_id': 'yanbaopptmerge_70a45bf10fb90ae0c91d44a11e0a97a8f9515d85626b54eb50a80b6146e5c2bf.pdf_10.jpg'}

"""














"""

{'gt_idx': [4], 
'gt': '[14]1:https://www.163.com/...\n[15]1:https://www.163.com/... 2:https://www.dianping.... 3：烤鱼行业品牌竞争分析.\n[16] 1：烤鱼行业竞争格局维度…．\n[17] 1:https://aiqicha.baidu.c... 2:上市公司信息',
 'pred_idx': [7], 
'pred': '[14] 1：https://www.163.com/...\n[15] 1：https://www.163.com/...\n[16] 1：https://www.dianping...\n[17] 1：https://aiqicha.baidu.com...\n2：上市公司信息', 
'edit': 0.2815533980582524,
'gt_position': [5], 
'pred_position': 540, 
'norm_gt': '141httpswww163com151httpswww163com2httpswwwdianping3烤鱼行业品牌竞争分析161烤鱼行业竞争格局维度171httpsaiqichabaiduc2上市公司信息',
'norm_pred': '141httpswww163com151httpswww163com161httpswwwdianping171httpsaiqichabaiducom2上市公司信息', 
'gt_category_type': 'text_block', 
'pred_category_type': 'text_block', 
'gt_attribute': [{'text_language': 'text_en_ch_mixed', 'text_background': 'single_colored', 'text_rotate': 'normal'}],
'img_id': 'yanbaopptmerge_70a45bf10fb90ae0c91d44a11e0a97a8f9515d85626b54eb50a80b6146e5c2bf.pdf_10.jpg'}, 

"""

"""
gt_table_list:
[
{'category_type': 'table',
'poly': [98.88907451996393, 485.43740433673287, 706.4349152824724, 485.43740433673287, 706.4349152824724, 634.749517744468, 98.88907451996393, 634.749517744468], 
'ignore': False, 
'order': 4, 
'anno_id': 5, 
'latex': '\\begin{tabular}{|l|l|l|l|} \\hline\n\\multicolumn{4}{l}{九毛九国际控股有限公司(09922)} \\\\ \\hline\n总市值 & 营收规模 & 同比增长(\\%) & 毛利率(\\%) \\\\ \\hline\n88.0亿 & 41.8亿 & 53.9600 & \\_ \\_ \\hline\\\\ \\hline\n\\end{tabular}',
'html': '<table>\n<thead>\n<tr>\n <th colspan="4" rowspan="2">九毛九国际控股有限公司(09922)</th>\n</tr>\n<tr>\n</tr>\n</thead>\n<tbody>\n<tr>\n <td>总市值</td>\n <td>营收规模</td>\n <td>同比增长(%)</td>\n <td>毛利率(%)</td>\n</tr>\n<tr>\n <td>88.0亿</td>\n <td>41.8亿</td>\n <td>53.9600</td>\n <td>_ _</td>\n</tr>\n</tbody>\n</table>', 
'attribute': {'table_layout': 'horizontal', 'with_span': True, 'line': 'wireless_line', 'language': 'table_simplified_chinese', 'include_equation': False, 'include_photo': False, 'include_background': True, 'with_structured_text': False}, 
'table_edit_status': 'good'},
{'category_type': 'table', 'poly': [74.32924890663881, 1433.6914006928678, 1553.1167323978748, 1433.6914006928678, 1553.1167323978748, 1933.505770406509, 74.32924890663881, 1933.505770406509], 'ignore': False, 'order': 12, 'anno_id': 10, 'latex': '\\begin{tabular}{|l|l|l|l|} \\hline\n\\multicolumn{4}{c}{·公司信息} \\\\ \\hline\n企业状态 & 存续 & 注册资本 & 13500万人民币 \\\\ \\hline\n企业总部 & 市辖区 & 行业 & 商务服务业 \\\\ \\hline\n法人 & 陈文豪 & 统一社会信用代码 & 91110000101443599G \\\\ \\hline\n企业类型 & 有限责任公司（台港澳法人独资） & 成立时间 & 1995-09-29 \\\\ \\hline\n品牌名称 & \\multicolumn{3}{l}{美诺（北京）餐饮管理有限公司} \\\\ \\hline\n经营范围 & \\multicolumn{3}{l}{代理记帐；以下项目限分支机构经营：餐饮服务（含凉菜、不含裱花蛋糕、不含生食水海产...查看更多}\\\\ \\hline\n\\end{tabular}\n', 'html': '<table>\n<thead>\n<tr>\n <th colspan="4">·公司信息</th>\n</tr>\n</thead>\n<tbody>\n<tr>\n <td>企业状态</td>\n <td>存续</td>\n <td>注册资本</td>\n <td>13500万人民币</td>\n</tr>\n<tr>\n <td>企业总部</td>\n <td>市辖区</td>\n <td>行业</td>\n <td>商务服务业</td>\n</tr>\n<tr>\n <td>法人</td>\n <td>陈文豪</td>\n <td>统一社会信用代码</td>\n <td>91110000101443599G</td>\n</tr>\n<tr>\n <td>企业类型</td>\n <td>有限责任公司（台港澳法人独资）</td>\n <td>成立时间</td>\n <td>1995-09-29</td>\n</tr>\n<tr>\n <td>品牌名称</td>\n <td colspan="3">美诺（北京）餐饮管理有限公司</td>\n</tr>\n<tr>\n <td>经营范围</td>\n <td colspan="3">代理记帐；以下项目限分支机构经营：餐饮服务（含凉菜、不含裱花蛋糕、不含生食水海产...查看更多</td>\n</tr>\n</tbody>\n</table>', 
'attribute': {'table_layout': 'horizontal', 'with_span': True, 'line': 'full_line', 'language': 'table_simplified_chinese', 'include_equation': False, 'include_photo': False, 'include_background': True, 'with_structured_text': False}, 
'table_edit_status': 'good'}
]
"""

"""
VLME

pred_content:(51,207),(476,257)

横纵名为门店数量，纵轴为毛利率。

(51,271),(476,321)

上市公司速览

(51,335),(476,385)

九毛九国际控股有限公司 (09922)

| 总市值 | 营收规模 | 同比增长(%) | 毛利率(%) |
| :--: | :--: | :--: | :--: |
| 88.0亿 | 41.8亿 | 53.9600 | -- |

(51,409),(476,459)

[14] 1：https://www.163.com/...
[15] 1：https://www.163.com/...
[16] 1：https://www.dianping...
[17] 1：https://aiqicha.baidu.com...
2：上市公司信息

(51,513),(476,563)

烤鱼代表企业分析

(51,587),(928,817)

美诺（北京）餐饮管理有限公司[18]

| 公司信息 |
| :--: |
| 企业状态 | 存续 |
| 企业总部 | 市辖区 |
| 法人 | 陈文豪 |
| 企业类型 | 有限责任公司(台港澳法人独资) |
| 品牌名称 | 美诺（北京）餐饮管理有限公司 |
| 经营范围 | 代理记帐；以下项目限分支机构经营：餐饮服务（含凉菜、不含裱花蛋糕、不含生食水海产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花


pred_dataset:defaultdict(<class 'list'>, 
{'text_all': [
{'category_type': 'text_all', 'position': [0, 18], 'content': '(51,207),(476,257)', 'fine_category_type': 'text_block'},
{'category_type': 'text_all', 'position': [18, 34], 'content': '横纵名为门店数量，纵轴为毛利率。', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [34, 52], 'content': '(51,271),(476,321)', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [52, 58], 'content': '上市公司速览', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [58, 76], 'content': '(51,335),(476,385)', 'fine_category_type': 'text_block'},
{'category_type': 'text_all', 'position': [76, 95], 'content': '九毛九国际控股有限公司 (09922)', 'fine_category_type': 'text_block'},
{'category_type': 'text_all', 'position': [95, 191], 'content': '| 总市值 | 营收规模 | 同比增长(%) | 毛利率(%) |\n| :--: | :--: | :--: | :--: |\n| 88.0亿 | 41.8亿 | 53.9600 | -- |', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [191, 209], 'content': '(51,409),(476,459)', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [209, 346], 'content': '[14] 1：https://www.163.com/...\n[15] 1：https://www.163.com/...\n[16] 1：https://www.dianping...\n[17] 1：https://aiqicha.baidu.com...\n2：上市公司信息', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [346, 364], 'content': '(51,513),(476,563)', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [364, 372], 'content': '烤鱼代表企业分析', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [372, 390], 'content': '(51,587),(928,817)', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [390, 408], 'content': '美诺（北京）餐饮管理有限公司[18]', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [408, 2811], 'content': '| 公司信息 |\n| :--: |\n| 企业状态 | 存续 |\n| 企业总部 | 市辖区 |\n| 法人 | 陈文豪 |\n| 企业类型 | 有限责任公司(台港澳法人独资) |\n| 品牌名称 | 美诺（北京）餐饮管理有限公司 |\n| 经营范围 | 代理记帐；以下项目限分支机构经营：餐饮服务（含凉菜、不含裱花蛋糕、不含生食水海产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花', 'fine_category_type': 'text_block'}
]
     })

"""

"""
omnidoc


pred_content:(51,207),(476,257)

横纵名为门店数量，纵轴为毛利率。

(51,271),(476,321)

上市公司速览

(51,335),(476,385)

九毛九国际控股有限公司 (09922)

| 总市值 | 营收规模 | 同比增长(%) | 毛利率(%) |
| :--: | :--: | :--: | :--: |
| 88.0亿 | 41.8亿 | 53.9600 | -- |

(51,409),(476,459)

[14] 1：https://www.163.com/...
[15] 1：https://www.163.com/...
[16] 1：https://www.dianping...
[17] 1：https://aiqicha.baidu.com...
2：上市公司信息

(51,513),(476,563)

烤鱼代表企业分析

(51,587),(928,817)

美诺（北京）餐饮管理有限公司[18]

| 公司信息 |
| :--: |
| 企业状态 | 存续 |
| 企业总部 | 市辖区 |
| 法人 | 陈文豪 |
| 企业类型 | 有限责任公司(台港澳法人独资) |
| 品牌名称 | 美诺（北京）餐饮管理有限公司 |
| 经营范围 | 代理记帐；以下项目限分支机构经营：餐饮服务（含凉菜、不含裱花蛋糕、不含生食水海产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花

pred_dataset:defaultdict(<class 'list'>, 
{'text_all': [{'category_type': 'text_all', 'position': [0, 19], 'content': '(51,207),(476,257)', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [19, 35], 'content': '横纵名为门店数量，纵轴为毛利率。', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [35, 53], 'content': '(51,271),(476,321)', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [53, 59], 'content': '上市公司速览', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [59, 77], 'content': '(51,335),(476,385)', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [77, 96], 'content': '九毛九国际控股有限公司 (09922)', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [522, 540], 'content': '(51,409),(476,459)', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [540, 677], 'content': '[14] 1：https://www.163.com/...\n[15] 1：https://www.163.com/...\n[16] 1：https://www.dianping...\n[17] 1：https://aiqicha.baidu.com...\n2：上市公司信息', 'fine_category_type': 'text_block'},
{'category_type': 'text_all', 'position': [677, 695], 'content': '(51,513),(476,563)', 'fine_category_type': 'text_block'}, 
{'category_type': 'text_all', 'position': [695, 703], 'content': '烤鱼代表企业分析', 'fine_category_type': 'text_block'},
 {'category_type': 'text_all', 'position': [703, 721], 'content': '(51,587),(928,817)', 'fine_category_type': 'text_block'},
   {'category_type': 'text_all', 'position': [721, 739], 'content': '美诺（北京）餐饮管理有限公司[18]', 'fine_category_type': 'text_block'}, 
   {'category_type': 'text_all', 'position': [1381, 3672], 'content': '| 经营范围 | 代理记帐；以下项目限分支机构经营：餐饮服务（含凉菜、不含裱花蛋糕、不含生食水海产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花蛋糕、不含生食水产品、不含裱花', 'fine_category_type': 'text_block'}
   ], 
'html_table': [{'category_type': 'html_table', 'position': [109, 534], 'content': '<table border="1" >\n    <tr>\n      <td colspan="1" rowspan="1">总市值</td>\n      <td colspan="1" rowspan="1">营收规模</td>\n      <td colspan="1" rowspan="1">同比增长(%)</td>\n      <td colspan="1" rowspan="1">毛利率(%)</td>\n    </tr>\n    <tr>\n      <td colspan="1" rowspan="1">88.0亿</td>\n      <td colspan="1" rowspan="1">41.8亿</td>\n      <td colspan="1" rowspan="1">53.9600</td>\n      <td colspan="1" rowspan="1">--</td>\n    </tr>\n</table>', 'fine_category_type': 'md2html_table'}, {'category_type': 'html_table', 'position': [768, 1409], 'content': '<table border="1" >\n    <tr>\n      <td colspan="1" rowspan="1">公司信息</td>\n    </tr>\n    <tr>\n      <td colspan="1" rowspan="1">企业状态</td>\n      <td colspan="1" rowspan="1">存续</td>\n    </tr>\n    <tr>\n      <td colspan="1" rowspan="1">企业总部</td>\n      <td colspan="1" rowspan="1">市辖区</td>\n    </tr>\n    <tr>\n      <td colspan="1" rowspan="1">法人</td>\n      <td colspan="1" rowspan="1">陈文豪</td>\n    </tr>\n    <tr>\n      <td colspan="1" rowspan="1">企业类型</td>\n      <td colspan="1" rowspan="1">有限责任公司(台港澳法人独资)</td>\n    </tr>\n    <tr>\n      <td colspan="1" rowspan="1">品牌名称</td>\n      <td colspan="1" rowspan="1">美诺（北京）餐饮管理有限公司</td>\n    </tr>\n</table>', 'fine_category_type': 'md2html_table'}]})
"""