import inspect
import json
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional

SCIENCE_MAP = {
    'chemistry_unit_consistency': ('unit_matching', 'evaluate_unit_consistency'),
    'physics_unit_consistency': ('unit_matching', 'evaluate_unit_consistency'),
    'geography_unit_consistency': ('unit_matching', 'evaluate_unit_consistency'),
    'biology_unit_consistency': ('unit_matching', 'evaluate_unit_consistency'),
    'material_unit_consistency': ('unit_matching', 'evaluate_unit_consistency'),
    'chemistry_molecular_format_validity': ('chemistry_format_validation', 'evaluate_molecular_format'),
    'chemistry_entity_option_constraint': ('options_matching', 'evaluate_options_constraint'),
    'chemistry_atom_count_constraint': ('chemistry_count_atom_checking', 'evaluate_atom_count'),
    'chemistry_atom_bond_constraint': ('chemistry_count_bond_checking', 'evaluate_bond_count'),
    'chemistry_atom_group_constraint': ('chemistry_count_group_checking', 'evaluate_group_count'),
    'chemistry_method_constraint': ('analysis_method_checking', 'evaluate_method_constraint'),
    'chemistry_reaction_steps_constraint': ('analysis_step_checking', 'evaluate_analysis_steps'),
    'chemistry_analysis_steps_constraint': ('analysis_step_checking', 'evaluate_analysis_steps'),
    'physics_method_constraint': ('analysis_method_checking', 'evaluate_method_constraint'),
    'physics_analysis_steps_constraint': ('analysis_step_checking', 'evaluate_analysis_steps'),
    'geography_method_constraint': ('analysis_method_checking', 'evaluate_method_constraint'),
    'geography_analysis_steps_constraint': ('analysis_step_checking', 'evaluate_analysis_steps'),
    'biology_method_constraint': ('analysis_method_checking', 'evaluate_method_constraint'),
    'biology_analysis_steps_constraint': ('analysis_step_checking', 'evaluate_analysis_steps'),
    'material_method_constraint': ('analysis_method_checking', 'evaluate_method_constraint'),
    'material_analysis_steps_constraint': ('analysis_step_checking', 'evaluate_analysis_steps'),
    'geography_address_format_validity': ('geography_format_geocoding_validation', 'evaluate_geography_address'),
    'geography_scene_option_constraint': ('options_matching', 'evaluate_options_constraint'),
    'biology_entity_relationship_format_validity':
    ('life_format_entity_relationship_validation', 'evaluate_entity_relationship'),
    'biology_sequence_length_constraint': ('life_sequence_length_checking', 'evaluate_sequence_length'),
    'material_characterization_technique_format_constraint':
    ('materials_format_characterization_technique_validation', 'evaluate_characterization_technique'),
    'material_property_prediction_constraint':
    ('materials_property_prediction_checking', 'evaluate_property_prediction'),
}

LEGACY_SCIENCE_ALIASES = {
    'life_unit_consistency':
    SCIENCE_MAP['biology_unit_consistency'],
    'life_method_constraint':
    SCIENCE_MAP['biology_method_constraint'],
    'life_analysis_steps_constraint':
    SCIENCE_MAP['biology_analysis_steps_constraint'],
    'life_entity_relationship_format_validity': (SCIENCE_MAP['biology_entity_relationship_format_validity']),
    'life_sequence_length_constraint':
    SCIENCE_MAP['biology_sequence_length_constraint'],
    'materials_unit_consistency':
    SCIENCE_MAP['material_unit_consistency'],
    'materials_method_constraint':
    SCIENCE_MAP['material_method_constraint'],
    'materials_analysis_steps_constraint':
    SCIENCE_MAP['material_analysis_steps_constraint'],
    'materials_characterization_technique_format_constraint':
    (SCIENCE_MAP['material_characterization_technique_format_constraint']),
    'materials_property_prediction_constraint': (SCIENCE_MAP['material_property_prediction_constraint']),
}

GENERAL_MAP = {
    'general_decimal_annotation': ('general_checking', 'check_decimal_format'),
    'general_scientific_annotation': ('general_checking', 'check_scientific_format'),
    'general_wrap_up': ('general_checking', 'check_wrap_up'),
    'general_all_uppercase': ('general_checking', 'check_uppercase'),
    'general_all_lowercase': ('general_checking', 'check_lowercase'),
    'general_json_constraint': ('general_checking', 'check_json_format'),
    'general_list_constraint': ('general_checking', 'check_list_format'),
    'general_tuple_constraint': ('general_checking', 'check_tuple_format'),
    'general_dictionary_constraint': ('general_checking', 'check_dictionary_format'),
    'general_markdown_constraint': ('general_checking', 'check_markdown_format'),
    'general_html_constraint': ('general_checking', 'check_html_format'),
    'general_xml_constraint': ('general_checking', 'check_xml_format'),
    'general_csv_constraint': ('general_checking', 'check_csv_format'),
    'general_choose_from': ('general_checking', 'check_choose_from'),
    'general_judge': ('general_checking', 'check_judge'),
    'general_number_response': ('general_checking', 'check_number_response'),
    'general_response_structure': ('general_checking', 'check_response_structure'),
}

INSTRUCTION_EVALUATOR_MAP = {
    **SCIENCE_MAP,
    **LEGACY_SCIENCE_ALIASES,
    **GENERAL_MAP,
}

LLM_EXTRACTION_INSTRUCTIONS = {
    'chemistry_molecular_format_validity',
    'chemistry_atom_count_constraint',
    'chemistry_atom_bond_constraint',
    'chemistry_atom_group_constraint',
    'chemistry_method_constraint',
    'physics_method_constraint',
    'geography_method_constraint',
    'biology_method_constraint',
    'material_method_constraint',
    'life_method_constraint',
    'materials_method_constraint',
}


def parse_instruction_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError('SciMIF instruction_list must contain valid JSON.') from exc
        if not isinstance(parsed, list):
            raise ValueError('SciMIF instruction_list must decode to a list.')
        return [item for item in parsed if isinstance(item, dict)]
    raise TypeError(f'Unsupported SciMIF instruction_list type: {type(value)!r}')


def get_evaluator(instruction_name: str):
    target = INSTRUCTION_EVALUATOR_MAP.get(instruction_name)
    if target is None:
        return None
    module_name, function_name = target
    module = import_module(f'.scimif.{module_name}', package=__package__)
    return getattr(module, function_name)


def evaluate_single_instruction(response: str,
                                item: Dict[str, Any],
                                instruction: Dict[str, Any],
                                llm_client=None,
                                judge_model: Optional[str] = None) -> Dict[str, Any]:
    instruction_name = instruction.get('instruction_name', '')
    evaluator = get_evaluator(instruction_name)
    if evaluator is None:
        return {
            'score': 0.0,
            'detail': f'Unsupported instruction: {instruction_name}',
            'skipped': True,
        }

    call_kwargs = {
        'required_parameters': instruction.get('required_parameters') or '',
        'instruction_description': '',
        'edit_question': item.get('edit_question', ''),
        'reference_answer': item.get('answer', ''),
        'item': item,
        'instruction_name': instruction_name,
    }
    if ('analysis_step' in instruction_name or 'reaction_steps' in instruction_name
            or instruction_name == 'general_number_response' or instruction_name in LLM_EXTRACTION_INSTRUCTIONS):
        call_kwargs['llm_client'] = llm_client
        call_kwargs['judge_model'] = judge_model

    try:
        signature = inspect.signature(evaluator)
        accepts_varkw = any(parameter.kind == inspect.Parameter.VAR_KEYWORD
                            for parameter in signature.parameters.values())
        if accepts_varkw:
            filtered_kwargs = call_kwargs
        else:
            filtered_kwargs = {key: value for key, value in call_kwargs.items() if key in signature.parameters}
        result = evaluator(response, **filtered_kwargs)
        if not isinstance(result, dict):
            return {
                'score': 0.0,
                'detail': f'Unexpected evaluator result type: {type(result)!r}',
                'skipped': False,
            }
        try:
            score = float(result.get('score', 0.0))
        except (TypeError, ValueError):
            score = 0.0
        return {
            **result,
            'score': min(1.0, max(0.0, score)),
            'detail': str(result.get('detail', '')),
            'skipped': bool(result.get('skipped', False)),
        }
    except Exception as exc:
        return {
            'score': 0.0,
            'detail': f'Evaluator error: {exc}',
            'skipped': False,
        }


def evaluate_record(item: Dict[str, Any], llm_client=None, judge_model: Optional[str] = None) -> Dict[str, Any]:
    response_value = item.get('prediction', item.get('response', ''))
    response = '' if response_value is None else str(response_value)
    instructions = parse_instruction_list(item.get('instruction_list'))
    results = []

    for instruction in instructions:
        evaluation = evaluate_single_instruction(
            response=response,
            item=item,
            instruction=instruction,
            llm_client=llm_client,
            judge_model=judge_model,
        )
        result = {
            'instruction_name': instruction.get('instruction_name', ''),
            'source': instruction.get('source', ''),
            'required_parameters': instruction.get('required_parameters') or '',
            'score': evaluation['score'],
            'detail': evaluation['detail'],
            'skipped': evaluation['skipped'],
        }
        if 'casing_scope' in evaluation:
            result['casing_scope'] = evaluation['casing_scope']
        results.append(result)

    evaluated = [result for result in results if not result['skipped']]
    instruction_score = (sum(result['score'] for result in evaluated) / len(evaluated) if evaluated else 0.0)
    strict_score = float(bool(evaluated) and all(result['score'] >= 1.0 for result in evaluated))

    return {
        'instruction_results': results,
        'instruction_score': instruction_score,
        'strict_score': strict_score,
        'evaluated_instructions': len(evaluated),
        'skipped_instructions': len(results) - len(evaluated),
    }


def summarize_results(records: Iterable[Dict[str, Any]],
                      subjects: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    record_list = list(records)
    if subjects is None:
        subject_names = sorted({str(record.get('subject', '')) for record in record_list})
    else:
        subject_names = list(subjects)

    groups = [('overall', record_list)]
    groups.extend((
        subject,
        [record for record in record_list if str(record.get('subject', '')) == subject],
    ) for subject in subject_names if subject)

    summary = []
    for group_name, group_records in groups:
        instruction_results = [
            result for record in group_records for result in record.get('instruction_results', [])
            if not result.get('skipped', False)
        ]
        instruction_score_sum = sum(float(result.get('score', 0.0)) for result in instruction_results)
        instruction_accuracy = instruction_score_sum / len(instruction_results) if instruction_results else 0.0
        sample_score_sum = sum(float(record.get('instruction_score', 0.0)) for record in group_records)
        sample_accuracy = sample_score_sum / len(group_records) if group_records else 0.0
        strict_score_sum = sum(float(record.get('strict_score', 0.0)) for record in group_records)
        strict_accuracy = strict_score_sum / len(group_records) if group_records else 0.0

        source_accuracy = {}
        for source in ('original', 'core_task', 'added_general'):
            source_results = [result for result in instruction_results if result.get('source', '') == source]
            source_score_sum = sum(float(result.get('score', 0.0)) for result in source_results)
            source_accuracy[f'{source}_accuracy'] = source_score_sum / len(source_results) if source_results else 0.0

        summary.append({
            'split': group_name,
            'samples': len(group_records),
            'instructions': len(instruction_results),
            'skipped': sum(int(record.get('skipped_instructions', 0)) for record in group_records),
            'instruction_accuracy': instruction_accuracy,
            'sample_accuracy': sample_accuracy,
            'strict_accuracy': strict_accuracy,
            **source_accuracy,
        })
    return summary
