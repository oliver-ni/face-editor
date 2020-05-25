const app = new Vue({
    el: "#app",
    data: {
        components: null,
        mean: null,
        scale: null,
        latent: null,
        mins: [...Array(300).keys()].map((i) => -3 * STDS[i]),
        maxs: [...Array(300).keys()].map((i) => 3 * STDS[i]),
    },
    watch: {
        latent: {
            handler(val) {
                if (!this.loaded) return;
                const latent = tf.tensor([val.map((e) => e.v)]);
                const rec = tf.dot(latent, this.components).mul(this.scale).add(this.mean);
                const prediction = this.model.predict(rec).reshape([64, 64, 3]);
                tf.browser.toPixels(prediction, this.$refs.canvas);
            },
            deep: true,
        },
    },
    computed: {
        loaded() {
            return this.components !== null && this.mean !== null && this.scale !== null;
        },
    },
    methods: {
        modelLoaded(components, mean, scale, model) {
            this.components = components;
            this.mean = mean;
            this.scale = scale;
            this.model = model;
            this.goToMean();
        },
        goToMean() {
            this.latent = [...Array(300).keys()].map((i) => ({ v: 0 }));
        },
        async generateRandom() {
            let latent = [];
            for (let i = 0; i < 300; i++) {
                latent.push(tf.randomNormal([1], 0, STDS[i]));
            }
            latent = tf.stack(latent, (axis = 1));
            this.latent = Array.from(await latent.data()).map((e) => ({ v: e }));
        },
    },
});

const loadNpy = (url) => new Promise((resolve) => NumpyLoader.ajax(url, resolve));

async function init() {
    const [components, mean, scale, model] = await Promise.all([
        loadNpy("components.npy"),
        loadNpy("mean.npy"),
        loadNpy("scale.npy"),
        tf.loadLayersModel("decoder/model.json"),
    ]);

    app.modelLoaded(
        tf.transpose(tf.tensor(components.data).reshape([300, 300])),
        tf.tensor(mean.data).reshape([300]),
        tf.tensor(scale.data).reshape([300]),
        model
    );
}

init();
