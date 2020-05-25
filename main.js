const app = new Vue({
    el: "#app",
    data: {
        components: null,
        mean: null,
        scale: null,
        latent: null,
        EXP: EXP,
        mins: [...Array(126).keys()].map((i) => MEAN[i] - 3 * STD[i]),
        maxs: [...Array(126).keys()].map((i) => MEAN[i] + 3 * STD[i]),
    },
    watch: {
        latent: {
            handler(val) {
                if (!this.loaded) return;
                const latent = tf.tensor([val.map((e) => e.v)]);
                const rec = tf.dot(latent, this.components).add(SHIFT);
                const prediction = this.model.predict(rec).reshape([64, 64, 3]);
                tf.browser.toPixels(prediction, this.$refs.canvas);
            },
            deep: true,
        },
    },
    computed: {
        loaded() {
            return this.components !== null && this.model !== null;
        },
    },
    methods: {
        modelLoaded(components, model) {
            this.components = components;
            this.model = model;
            this.goToMean();
        },
        goToMean() {
            this.latent = MEAN.map((v) => ({ v }));
        },
        async generateRandom() {
            let latent = [];
            for (let i = 0; i < 126; i++) {
                latent.push(tf.randomNormal([1], MEAN[i], STD[i]));
            }
            latent = tf.stack(latent, (axis = 1));
            this.latent = Array.from(await latent.data()).map((v) => ({ v }));
        },
    },
});

const loadNpy = (url) => new Promise((resolve) => NumpyLoader.ajax(url, resolve));

async function init() {
    const [components, model] = await Promise.all([
        loadNpy("components.npy"),
        tf.loadLayersModel("decoder/model.json"),
    ]);

    app.modelLoaded(tf.transpose(tf.tensor(components.data).reshape([300, 126])), model);
}

init();
