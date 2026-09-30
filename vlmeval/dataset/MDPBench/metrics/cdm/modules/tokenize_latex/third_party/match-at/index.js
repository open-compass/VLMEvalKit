"use strict";

function getFlags(regex) {
    var flags = "";

    if (regex.ignoreCase) {
        flags += "i";
    }
    if (regex.multiline) {
        flags += "m";
    }
    if (regex.unicode) {
        flags += "u";
    }
    if (regex.dotAll) {
        flags += "s";
    }

    return flags;
}

module.exports = function matchAt(regex, input, position) {
    if (!(regex instanceof RegExp)) {
        throw new TypeError("Expected regex to be a RegExp");
    }

    var anchored = new RegExp("^(?:" + regex.source + ")", getFlags(regex));
    return anchored.exec(input.slice(position));
};
