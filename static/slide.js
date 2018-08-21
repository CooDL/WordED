/* var doscroll = function(){
     var $parent = $('.js-slide-list');
     var $first = $parent.find('li:first');
     var height = $first.height();
     $first.animate({
         marginTop: -height + 'px'
         }, 500, function() {
         $first.css('marginTop', 0).appendTo($parent);
     });    
};
setInterval(function(){doscroll()}, 2000);
*/

var doscroll = function(){
     var $parent = $('.table-slide');
     var $first = $parent.find('tr:first');
     var height = $first.height();
     $first.animate({
         marginTop: -height + 'px'
         }, 500, function() {
         $first.css('marginTop', 0).appendTo($parent);
     });    
};
setInterval(function(){doscroll()}, 1000);

var cluscroll = function(){
     var $parent = $('.cluster-slide');
     var $first = $parent.find('tr:first');
     var height = $first.height();
    
     $first.animate({
         marginTop: -height + 'px'
         }, 500, function() {
         $first.css('marginTop', 0).appendTo($parent);
     });    
    
};
setInterval(function(){cluscroll()}, 1000);
