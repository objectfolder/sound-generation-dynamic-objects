# Sound Generation of Dynamic Objects

Given a video clip of a falling object, the goal of this task is to generate the corresponding sound based on the visual appearance and motion of the object. The generated sound must match the object’s intrinsic properties (e.g., material type) and temporally align with the object’s movement in the given video. This task is related to prior work on sound generation from in-the-wild videos, but here we focus more on predicting soundtracks that closely match the object dynamics.

## Usage

#### Training & Evaluation

Start the training process, and test the best model on test-set after training:

```sh
python main.py --batch_size 32 --weight_decay 1e-2 --lr 1e-3 \
               --model RegNet --exp RegNet \
               --config_location ./configs/regnet_aux_4.yml
```

Evaluate the best model in *RegNet*:

```sh
python main.py --batch_size 32 --weight_decay 1e-2 --lr 1e-3 \
               --model RegNet --exp RegNet \
               --config_location ./configs/regnet_aux_4.yml \
               --eval
```

#### Add your own model

To train and test your new model on ObjectFolder Sound Generation of Dynamic Objects Benchmark, you only need to modify several files in *models*, you may follow these simple steps.

1. Create new model directory

    ```sh
    mkdir models/my_model
    ```

2. Design new model

    ```sh
    cd models/my_model
    touch my_model.py
    ```

3. Build the new model and its optimizer

    Add the following code into *models/build.py*:

    ```python
    elif args.model == 'my_model':
        from my_model import my_model
        model = my_model.my_model(args)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ```

4. Add the new model into the pipeline

    Once the new model is built, it can be trained and evaluated similarly:

    ```sh
    python main.py --batch_size 32 --weight_decay 1e-2 --lr 1e-3 \
                   --model my_model --exp my_model \
                   --config_location ./configs/my_model.yml
    ```

## Results on ObjectFolder Sound Generation of Dynamic Objects Benchmark

In our experiments, we choose 500 objects with reasonable scales, and 10 videos are generated for each object. We split the 10 videos into train/val/test splits of 8/1/1.


#### Results on ObjectFolder

<table>
    <tr>
        <td>Method</td>
        <td>STFT</td>
        <td>Envelope</td>
        <td>CDPAM</td>
    </tr>
    <tr>
        <td>RegNet</td>
        <td>0.010</td>
        <td>0.036</td>
        <td>0.0000565</td>
    </tr>
  	<tr>
        <td>MCR</td>
        <td>0.034</td>
        <td>0.042</td>
        <td>0.0000592</td>
    </tr>
</table>

