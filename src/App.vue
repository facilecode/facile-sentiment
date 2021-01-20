<template>
  <v-app>
    <!-- <v-app-bar app color="primary" dark>
    </v-app-bar> -->

    <v-container>
      <v-row align="center" justify="center">
        <v-col >
          <v-card class="mx-auto" max-width="100" >
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
            <v-text-field label="Your Text" v-model="text">
            </v-text-field>
            RES - {{res}}
          </div>
        </v-col>
        <!-- end-filter -->
        <v-col>
          <v-card class="mx-auto" max-width="100">
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
    res: [999, 999],
    text: '',
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
  computed: {
    res: function () {
      console.log(this.text)
      return this.res
    }
  },
  async mounted () {
    // test replace
    const text = 'T$his *is -a _t!ext wiéth so@me sp)ecia-l- char\'act"er\'s'
    const tokens = this.sentenceTokenizer(text)
    console.log('Tokens: ', tokens)
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
      this.res[1] = out.values().next().value.data
    },
    async loadTF () {
      this.tfModel = await tf.loadLayersModel('tfjs-model/model.json')
      // var data = tf.tensor1d([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      var data = tf.ones([1, 50])
      data.print()
      const out = this.tfModel.predict(data)
      console.log('tf out', out.dataSync())
      this.res[0] = out.dataSync()
    },
    sentenceTokenizer (sentence) {
      const filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'.split('')
      filter.map(f => {
        sentence = sentence.replaceAll(f, '')
      })
      return sentence.split(' ')
    }
  }
}
</script>
