<template>
  <v-app>
    <!-- <v-app-bar app color="primary" dark>
    </v-app-bar> -->

    <v-container>
      <v-row align="center" justify="center">
        <v-col >
          <v-card class="mx-auto" max-width="400" >
              <v-img src="./assets/pt-logo.png"></v-img>

              <v-card-title>
                  Title
              </v-card-title>
              <v-card-subtitle>
                  Subtitle
              </v-card-subtitle>

              <v-card-actions>
              <v-btn color="orange lighten-2" text > Explore </v-btn>
              <v-spacer></v-spacer>
              <v-btn color="orange lighten-2" text >Load Image</v-btn>
              </v-card-actions>
          </v-card>
        </v-col>
        <!-- input fields -->
        <v-col>
          <div>
            <!-- <v-text-field label="Your Text" :rules="rules"> -->
            <v-text-field label="Your Text">
            </v-text-field>
          </div>
        </v-col>
        <!-- end-filter -->
        <v-col>
          <v-card class="mx-auto" max-width="400">
            <v-img src="./assets/tf.png"></v-img>

            <v-card-title>
                Title
            </v-card-title>
            <v-card-subtitle>
                Subtitle
            </v-card-subtitle>

            <v-card-actions>
            <v-btn color="orange lighten-2" text > Explore </v-btn>
            <v-spacer></v-spacer>
            <v-btn color="orange lighten-2" text >Load Image</v-btn>
            </v-card-actions>
          </v-card>
        </v-col>
      </v-row>
    </v-container>

  </v-app>
</template>

<script>
// import Main from './views/Main.vue'
import { InferenceSession, Tensor } from 'onnxjs'
import * as tf from '@tensorflow/tfjs'

export default {
  name: 'App',

  components: {
  },

  data: () => ({
    maxlength: 50,
    rules: [
      value => (value && value.length < this.maxlength)
    ],
    // onnx models
    onnxModelFile: null,
    onnxSession: null,
    onnxModel: null,
    // tf model
    tfModel: null
  }),
  async mounted () {
    await this.loadONNX()
    console.log('ONNX loaded')
    await this.loadTF()
    console.log('TensorFlow JS loaded')
  },
  methods: {
    async loadONNX () {
      const res = await fetch('model.onnx')
      this.onnxModelFile = await res.arrayBuffer()
      this.onnxSession = new InferenceSession()
      await this.onnxSession.loadModel(this.onnxModelFile)
      // map-word-to-index()
      // var inputs = [new Tensor(new Int32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 'int32', [1, 10])]
      // var inputs = [new Tensor(new Int32Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'int32', [1, 10])]
      var inputs = [new Tensor(new Int32Array([2, 9, 7, 4, 6, 5, 8, 1, 8, 2]), 'int32', [1, 10])]
      const out = await this.onnxSession.run(inputs)
      console.log('ONNX out ', out.values().next().value.data)
    },
    async loadTF () {
      this.tfModel = await tf.loadLayersModel('tfjs-model/model.json')
      // var data = tf.tensor1d([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      var data = tf.ones([1, 50])
      data.print()
      const out = this.tfModel.predict(data)
      out.print()
    }
  }
}
</script>
