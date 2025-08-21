from process_data import ProcessData
from dash import Dash, html


class CreateDashboard():
    def __init__(self):
        self.process_data = ProcessData
        self.stacked_line_data = self.process_data()
        print(self.stacked_line_data)
    
app = Dash()

app.layout = [html.Div(children='Hello World')]

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)


        