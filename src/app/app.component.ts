import { Component } from '@angular/core';
import * as tf from '@tensorflow/tfjs';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  linearModel: tf.Sequential;
  prediction: any;
  title = 'app';

  ngOnInit() {
    this.train();
  }


  async train() {
    this.linearModel =tf.sequential();
    this.linearModel.add(tf.layers.dense({units:1, inputShape:[1]}));
    this.linearModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});


  const xs = tf.tensor1d([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]);
  const ys = tf.tensor1d([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]);


  await this.linearModel.fit(xs, ys)

  console.log('Finished training model..')
  }

  predict(val: number) {
  const output = this.linearModel.predict(tf.tensor2d([val], [1, 1])) as any;
  this.prediction = Array.from(output.dataSync())[0]
}
}
