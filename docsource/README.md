# Generating the documentation


## Packages installed

Building it requires the package `sphinx` that you can
install using:

```bash
pip install -r requirements.txt
```

## Building the documentation

Once you have setup `sphinx`, you can build the documentation by running the following command in the `/docs` folder:

```bash
make html
```

A folder called ``build/html`` should have been created. You can now open the file ``build/html/index.html`` in your
browser. 

---
**NOTE**

If you are adding/removing elements from the toc-tree or from any structural item, it is recommended to clean the build
directory before rebuilding. Run the following command to clean and build:

```bash
make clean && make html
```

---

It should build the static app that will be available under `/docs/build/html`
