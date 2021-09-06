# Aloception - Development

## Building the documentation

See <a href="./docs/README.md"> docs/README.md</a>

## Useful extension
- [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring). In setting, change the format to `numpy`

## Unit test

```
python -m pytest
```

NOTE: You might want to use only one GPU to run the tests (Multi-GPU is gonna be used insteads to test the training pipelines.)


## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

See the [open issues](https://gitlab.com/aigen-vision/aloception/issues) for a list of proposed features (and known issues).

..TODO..

## Submitting & merging merge request

- Ideally, all merge request should be linked with a specific issue.
- If the merge request is linked with an issue, the name should be as follow: #issueID-name-of-mg with the list of Issue to automatically close once the merge request is merged:
(closes #1, closes#2)
- A merge request must include the list of all changes & added features.
- The reviewer is added only once the merge request is ready to be merged.
Once the reviewer finished reviewing the merge request is `approved` or not `approved` (using the approved button) along with a review. Then, the reviewer should remove itself from the merge request.
If `approved` : the MR can be merge.
Otherwise : All thread of the review must be resolved before adding back a reviewer on the MR.
## License

TODO 
