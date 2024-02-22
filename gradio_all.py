import os

import gradio as gr



if __name__ == "__main__":
    global_holder = {}

    with gr.Blocks() as demo:
        gr.Markdown("Face Swapping All In One")

        with gr.Tab("Image"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    image1_input = gr.Image(label="source")
                    image2_input = gr.Image(label="target")
                with gr.Column(scale=2):
                    image_output = gr.Image(label="result")
                    image_button = gr.Button("Run")

        # with gr.Tab("Video"):
        #     with gr.Row(equal_height=True):
        #         with gr.Column(scale=2):
        #             image3_input = gr.Image(label="source image")
        #             video_input = gr.Video(label="target video")
        #         with gr.Column(scale=3):
        #             video_output = gr.Video(label="result")
        #             video_button = gr.Button("Run")
        #     with gr.Accordion("Advanced Video Swapping Options", open=False):
        #         frames_cnt = gr.Slider(label="Target Max Frames Count (-1: use all frames)",
        #                                minimum=-1, maximum=9999, value=-1, step=1)
        #         use_crop = gr.Checkbox(label='Crop Inputs? (crop and align the faces)', value=True)
        #         use_pti = gr.Checkbox(
        #             label='Enable PTI Tuning (finetuning the generator to obtain more stable video result',
        #             value=True)
        #     with gr.Accordion("Advanced PTI Tuning Options", open=False):
        #         pti_steps = gr.Slider(label="Max PTI Steps", minimum=0, maximum=999, value=80, step=1)
        #         pti_lr = gr.Slider(label="PTI Learning Rate", minimum=0.0, maximum=1e-1, value=1e-3, step=0.0001)
        #         pti_recolor_lambda = gr.Slider(label="Recolor Lambda", minimum=0.0, maximum=20.0, value=5.0, step=0.1)
        #         pti_resume_weight_path = gr.Textbox(label="PTI Resume Weight Path",
        #                                             value='/Your/Path/To/PTI_G_lr??_iters??.pth')

        image_button.click(
            swap_image_gr,
            inputs=[image1_input, image2_input],
            outputs=image_output,
        )
        # video_button.click(
        #     swap_video_gr,
        #     inputs=[image3_input, video_input,
        #             frames_cnt, use_crop, use_pti,
        #             pti_steps, pti_lr, pti_recolor_lambda, pti_resume_weight_path,
        #             ],
        #     outputs=video_output,
        # )

    demo.launch(server_name="0.0.0.0", server_port=7868)
