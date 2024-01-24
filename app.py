from vlmeval.smp import *
import gradio as gr

with gr.Blocks() as demo:
    struct = load_results()
    timestamp = struct['time']
    EVAL_TIME = format_timestamp(timestamp)
    results = struct['results']
    N_MODEL = len(results)
    N_DATA = len(results['LLaVA-v1.5-7B']) - 1
    DATASETS = list(results['LLaVA-v1.5-7B'])
    DATASETS.remove('META')
    print(DATASETS)

    gr.Markdown(LEADERBORAD_INTRODUCTION.format(N_MODEL, N_DATA, EVAL_TIME))
    structs = [abc.abstractproperty() for _ in range(N_DATA)]

    with gr.Tabs(elem_classes='tab-buttons') as tabs:
        with gr.TabItem('🏅 OpenVLM Main Leaderboard', elem_id='main', id=0):
            gr.Markdown(LEADERBOARD_MD['MAIN'])
            table, check_box = BUILD_L1_DF(results, MAIN_FIELDS)
            type_map = check_box['type_map']
            checkbox_group = gr.CheckboxGroup(
                choices=check_box['all'],
                value=check_box['required'],
                label="Evaluation Dimension",
                interactive=True,
            )
            headers = check_box['essential'] + checkbox_group.value
            with gr.Row():
                model_size = gr.CheckboxGroup(
                    choices=MODEL_SIZE, 
                    value=MODEL_SIZE, 
                    label='Model Size',
                    interactive=True
                )
                model_type = gr.CheckboxGroup(
                    choices=MODEL_TYPE, 
                    value=MODEL_TYPE, 
                    label='Model Type',
                    interactive=True
                )
            data_component = gr.components.DataFrame(
                value=table[headers], 
                type="pandas", 
                datatype=[type_map[x] for x in headers],
                interactive=False, 
                visible=True)
            
            def filter_df(fields, model_size, model_type):
                headers = check_box['essential'] + fields
                df = cp.deepcopy(table)
                df['flag'] = [model_size_flag(x, model_size) for x in df['Parameters (B)']]
                df = df[df['flag']]
                df.pop('flag')
                if len(df):
                    df['flag'] = [model_type_flag(df.iloc[i], model_type) for i in range(len(df))]
                    df = df[df['flag']]
                    df.pop('flag')
                
                comp = gr.components.DataFrame(
                    value=df[headers], 
                    type="pandas", 
                    datatype=[type_map[x] for x in headers],
                    interactive=False, 
                    visible=True)
                return comp

            for cbox in [checkbox_group, model_size, model_type]:
                cbox.change(fn=filter_df, inputs=[checkbox_group, model_size, model_type], outputs=data_component)

        with gr.TabItem('🔍 About', elem_id='about', id=1):
            gr.Markdown(urlopen(VLMEVALKIT_README).read().decode())

        for i, dataset in enumerate(DATASETS):
            with gr.TabItem(f'📊 {dataset} Leaderboard', elem_id=dataset, id=i + 2):
                if dataset in LEADERBOARD_MD:
                    gr.Markdown(LEADERBOARD_MD[dataset])

                s = structs[i]
                s.table, s.check_box = BUILD_L2_DF(results, dataset)
                s.type_map = s.check_box['type_map']
                s.checkbox_group = gr.CheckboxGroup(
                    choices=s.check_box['all'],
                    value=s.check_box['required'],
                    label=f"{dataset} CheckBoxes",
                    interactive=True,
                )
                s.headers = s.check_box['essential'] + s.checkbox_group.value
                with gr.Row():
                    s.model_size = gr.CheckboxGroup(
                        choices=MODEL_SIZE, 
                        value=MODEL_SIZE, 
                        label='Model Size',
                        interactive=True
                    )
                    s.model_type = gr.CheckboxGroup(
                        choices=MODEL_TYPE, 
                        value=MODEL_TYPE, 
                        label='Model Type',
                        interactive=True
                    )
                s.data_component = gr.components.DataFrame(
                    value=s.table[s.headers], 
                    type="pandas", 
                    datatype=[s.type_map[x] for x in s.headers],
                    interactive=False, 
                    visible=True)
                s.dataset = gr.Textbox(value=dataset, label=dataset, visible=False)
                
                def filter_df_l2(dataset_name, fields, model_size, model_type):
                    s = structs[DATASETS.index(dataset_name)]
                    headers = s.check_box['essential'] + fields
                    df = cp.deepcopy(s.table)
                    df['flag'] = [model_size_flag(x, model_size) for x in df['Parameters (B)']]
                    df = df[df['flag']]
                    df.pop('flag')
                    if len(df):
                        df['flag'] = [model_type_flag(df.iloc[i], model_type) for i in range(len(df))]
                        df = df[df['flag']]
                        df.pop('flag')
                    
                    comp = gr.components.DataFrame(
                        value=df[headers], 
                        type="pandas", 
                        datatype=[s.type_map[x] for x in headers],
                        interactive=False, 
                        visible=True)
                    return comp

                for cbox in [s.checkbox_group, s.model_size, s.model_type]:
                    cbox.change(fn=filter_df_l2, inputs=[s.dataset, s.checkbox_group, s.model_size, s.model_type], outputs=s.data_component)
        

    with gr.Row():
        with gr.Accordion("Citation", open=False):
            citation_button = gr.Textbox(
                value=CITATION_BUTTON_TEXT, 
                label=CITATION_BUTTON_LABEL,
                elem_id='citation-button')

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0')