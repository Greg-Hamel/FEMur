class Mesh1D(Mesh):
    '1 Dimensional Mesh.'
    def __init__(self, domain, Number_of_elements, Nodes_per_element,
                 conditions):
        self.start = domain[0]  # First 'x' of the domain
        self.end = domain[1]  # Last 'x' of the domain
        self.num_elements = Number_of_elements  # Number of elements
        self.nodes_elements = Nodes_per_element  # Number of nodes per element
        self.d = np.array(conditions)  # Conditions at the nodes (d vector)

        self.length = self.end - self.start      # Length of the domain
        self.num_nodes = (
                          self.num_elements
                          + 1
                          + self.num_elements * (self.nodes_elements - 2)
                          )
        self.L_element = self.length / self.num_elements
        self.node_distance = self.L_element / (self.nodes_elements - 1)

        self.meshing = None
        self.Le_container = None
        self.N = None
        self.calculated = False  # True when elements trials are calculated

    def __str__(self):
        # Prints the element-node interaction.
        if self.meshing is None:
            Err = 'Error, the one-dimension mesh has not been generated yet. '\
                  'Please use Mesh1D.mesh() to generate your mesh based on '\
                  'values provided to Mesh1D.'
            return Err

        else:
            n_nodes = len(self.meshing[0])  # Checks the new created values
            n_elements = len(self.meshing[1])  # Checks the new created values
            output = f'The mesh contains {n_nodes} nodes and {n_elements} '\
                     'elements.'
            print(output)

            str_output = ""
            for i in range(n_elements):
                str_output = str_output + str(self.meshing[1][str(i)]) + "\n"

            return str_output

    def mesh(self):
        # Create a mesh for linear models
        nodes = {}
        elements = {}

        for i in range(self.num_nodes):  # Nodes Creation
            nodes[str(i)] = (
                Node1D((self.node_distance * i) + self.start, i)
                )

        for i in range(self.num_elements):  # Elements Creation
            element_nodes = range(
                        i * (self.nodes_elements - 1),
                        i * (self.nodes_elements - 1) + self.nodes_elements,
                        1
                        )
            nodes_table = []
            for j in list(element_nodes):
                nodes_table.append(nodes[str(j)])

            elements[str(i)] = Element1D(nodes_table)

        self.nodes = nodes
        self.elements = elements
        self.meshing = [nodes, elements]
        return nodes, elements

    def get_Le_container(self):
        # Get the dictionary which contains all L^e matrices
        Le = {}
        for i in self.elements.keys():
            Le[i] = np.zeros((self.nodes_elements, self.num_nodes))
            for j in range(self.nodes_elements):
                Le[i][j, (int(i) * (self.nodes_elements - 1)) + j] = 1

        self.Le_container = Le

    def get_de_container(self):
        # Get the dictionary which contains all d^e matrices
        de = {}
        if self.Le_container is None:
            self.get_Le_container()

        for i in self.elements.keys():
            de[i] = np.dot(self.Le_container[i], self.d)

        self.de = de

    def solve_elements(self):
        # Solve all current elements (shape functions, approximation, etc)
        self.get_de_container()
        for i in self.elements.keys():
            key = int(i)
            print(f"Calculating Element({key})'s shape functions")
            self.elements[i].get_Ne()

            validation = self.elements[i].validate_Ne()
            print(f'Validation of shape function is: {validation}')

            print(f"Calculating Element({key})'s shape functions derivatives")
            self.elements[i].get_Be()

            print(f"Injecting Conditions to Element({key})'s Shape Functions")
            self.elements[i].set_conditions(self.de[i])

            print(f"Calculating Element({key})'s trial functions")
            self.elements[i].get_trial()

            print(f"Calculating Element({key})'s trial derivative functions\n")
            self.elements[i].get_trial_prime()

            self.calculated = True

    def print_elements_trial(self):
        # Shows each element's trial function
        for i in self.elements.keys():
            if self.elements[i].trial is None:
                self.solve_elements()

            key = int(i)
            print(f'Element({key}) has a trial function of: '
                  f'{self.elements[i].trial}')

    def print_elements_Ne(self):
        # Shows each element's trial function
        for i in self.elements.keys():
            if self.elements[i].Ne is None:
                self.elements[i].get_Ne()

            key = int(i)
            print(f'Element({key}) has a trial function of: '
                  f'{self.elements[i].Ne}')

    def get_N(self):
        # Get the global shape function matrix (N)
        N = [0] * self.num_nodes

        if self.Le_container is None:
            self.get_Le_container()

        self.solve_elements()
        print("Calculating Global Shape functions")
        for i in range(self.num_nodes):
            for j in self.elements.keys():
                N[i] = N[i] + np.dot(self.elements[j].Ne,
                                     self.Le_container[j])[i]

        self.N = N

    def get_trial(self):
        # Get the global trial function
        # Warning: this does not seem to work properly.
        if self.N is None:
            self.get_N()

        omega = np.dot(self.N, self.d)

        self.omega = omega

    def get_global_weights(self):
        # Get the global weight function (omega)
        omega = np.zeros(self.num_elements)

        if self.Le_container is None:
            get_Le_container()

        for i in self.elements.keys():
            omega[int(i)] = np.dot(self.elements[i].Be,
                                   self.Le_container[i])

        self.omega = np.dot((np.sum(omega)), self.d)

    def get_trial_L2(self, expected):
        # Get the L2 norm error for the given expected function vs FEM results
        integral = 0
        for i in self.elements.keys():
            xi = sy.symbols('xi')
            expr_exp = sy.sympify(expected)
            expr_approx = self.elements[i].trial

            expr_error = (expr_exp - expr_approx) ** 2

            domain = [self.elements[i].start, self.elements[i].end]
            order = sy.degree(expr_error, x)

            length = domain[-1] - domain[0]
            npg = ceil((order + 1) / 2)

            new_x = (0.5 * (domain[0] + domain[1])
                     + 0.5 * xi * (domain[1] - domain[0]))
            expr = expr_error.subs(x, new_x)

            [new_xi, w] = p_roots(npg)

            for j in range(len(new_xi)):
                integral = (integral
                            + (w[j] * length * 0.5 * expr.subs(xi,
                                                               new_xi[j]))
                            )

        print(integral)
        self.L2_error = integral

    def get_trial_derivative_L2(self, expected):
        # Get the L2 norm error for the given expected function vs FEM results
        integral = 0
        for i in self.elements.keys():
            xi = sy.symbols('xi')
            expr_exp = sy.sympify(expected)
            expr_approx = self.elements[i].trial_prime

            expr_error = (expr_exp - expr_approx) ** 2

            domain = [self.elements[i].start, self.elements[i].end]
            order = sy.degree(expr_error, x)

            length = domain[-1] - domain[0]
            npg = ceil((order + 1) / 2)

            new_x = (0.5 * (domain[0] + domain[1])
                     + 0.5 * xi * (domain[1] - domain[0]))
            expr = expr_error.subs(x, new_x)

            [new_xi, w] = p_roots(npg)

            for j in range(len(new_xi)):
                integral = (integral
                            + (w[j] * length * 0.5 * expr.subs(xi,
                                                               new_xi[j]))
                            )

        print(integral)
        self.L2_error = integral

    def plot_trial_comparison(self, real_equation):
        if self.calculated is False:
            self.solve_elements()
        plot_x = np.linspace(self.start, self.end)
        equation = sy.sympify(real_equation)
        equation = sy.lambdify(x, equation)

        fem = []
        for i in self.elements.keys():
            f = sy.lambdify(x, self.elements[i].trial)
            for j in plot_x:
                if j >= self.elements[i].start and j <= self.elements[i].end:
                    if i == '0' and j == self.elements[i].start:
                        fem.append(f(j))
                    elif i != '0' and j == self.elements[i].start:
                        pass
                    else:
                        fem.append(f(j))
                else:
                    pass

        plt.plot(plot_x, equation(plot_x), 'b-', plot_x, fem, 'r--')
        plt.show()

    def plot_trial_derivative_comparison(self, real_equation):
        if self.calculated is False:
            self.solve_elements()

        plot_x = np.linspace(self.start, self.end)
        equation = sy.sympify(real_equation)
        equation = sy.lambdify(x, equation)

        fem = []
        for i in self.elements.keys():
            f_prime = sy.lambdify(x, self.elements[i].trial_prime)
            for j in plot_x:
                if j >= self.elements[i].start and j <= self.elements[i].end:
                    if i == '0' and j == self.elements[i].start:
                        fem.append(f_prime(j))
                    elif i != '0' and j == self.elements[i].start:
                        pass
                    else:
                        fem.append(f_prime(j))
                else:
                    pass

        plt.plot(plot_x, equation(plot_x), 'b-', plot_x, fem, 'r--')
        plt.show()
