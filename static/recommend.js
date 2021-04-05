$(function() {
  // Button will be disabled until we type anything inside the input field
  const source = document.getElementById('autoComplete');
  const inputHandler = function(e) {
    if(e.target.value==""){
      $('.product-button').attr('disabled', true);
    }
    else{
      $('.product-button').attr('disabled', false);
    }
  }
  source.addEventListener('input', inputHandler);

  $('.product-button').on('click',function(){
    var title = $('.movie').val();
    if (title=="") {
      $('.results').css('display','none');
      $('.fail').css('display','block');
    }
    else{
      load_details(title);
    }
  });
});

// get the basic details of the movie from the API (based on the name of the movie)
function load_details(title){
  $.ajax({
    type: 'GET',
    url:'/recommend?userid='+title,
    dataType: 'html',
    complete: function(){
      $("#loader").delay(500).fadeOut();
    },
    success: function(results){
      if(results.length<1){
        $('.fail').css('display','block');
        $('.results').css('display','none');
        $("#loader").delay(500).fadeOut();
      }
      else{
        $("#loader").fadeIn();
        $('.fail').css('display','none');
        $('.results').delay(1000).css('display','block');
        $('.results').html(results);
        $('#autoComplete').val('');
        $(window).scrollTop(0);
      }
    },
    error: function(){
      alert('Invalid Request');
      $("#loader").delay(500).fadeOut();
    },
  });
}
