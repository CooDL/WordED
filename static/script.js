
function changeAColor(link) {

    var alllinks = document.getElementById("left").getElementsByTagName("a");
    for (var i = 0; i < alllinks.length; i++) {
        alllinks[i].parentNode.className = "left-button-sub";
    }
    link.parentNode.className = "left-button-current";
}

