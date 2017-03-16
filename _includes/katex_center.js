<script>
    txlist = document.getElementsByTagName("dtex");
    for (var i = 0; i < txlist.length; i++) {
        var tx = txlist[i];
        var txtext = "\\displaystyle " + tx.textContent;
        var html = katex.renderToString(txtext, tx, { displayMode: true });
        html = "<div class='equation'>" + html 
                   + "<span style='float:right'>(" + (i+1) + ")</span></div>";
        tx.innerHTML = html;
    }
</script>
