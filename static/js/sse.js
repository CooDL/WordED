function sse() {
    var source = new EventSource('/stream_t');
    var out    = document.getElementsByClassName('table-slide');
    source.onmessage = function(e) {
        out.innerHTML = out.innerHTML + '\n' + '<tr><td></td> <td>'+e.data+'</td></tr>';
    };
}
$('#button').click(function(){
    alert('clicked!');
    $.post('/post_t/', {'message': $('#begin_time').val()});
    //$(this).val('');
});

sse();
