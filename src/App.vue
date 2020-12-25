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

export default {
  name: 'App',

  components: {
  },

  data: () => ({
    //
  })
}
</script>
