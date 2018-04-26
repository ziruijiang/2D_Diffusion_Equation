import numpy as np
def it_gauss_sedidel(A, b, x, e):
    '''
    Ax = b with n iterations until error < e
    '''
    solutions = []
    L = np.tril(A)
    n = 1
    x = np.dot(np.linalg.inv(L), b - np.dot(A-L, x))
    err = 1
    solutions.append(list(x))    
    while err>e:
        x = np.dot(np.linalg.inv(L), b - np.dot(A-L, x))
        solutions.append(list(x))  
        err = np.linalg.norm(x-solutions[-2])/np.linalg.norm(x)
        n+=1
    return n, solutions[-1]

def it_jacobi(A, b, x, e):
    '''
    Ax = b with n iterations until error < e
    '''
    solutions = []
    D = np.diag(A)
    P = A - np.diagflat(D)
    n = 1
    x = (b - np.dot(P,x))/ D
    err = 1
    solutions.append(list(x))
    while err > e:
        x = (b - np.dot(P,x))/ D
        solutions.append(list(x))        
        err = np.linalg.norm(x-solutions[-2])/np.linalg.norm(x)
        n += 1
    return n, solutions[-1]

def it_sor(A, b, x, e, w = 1.1):
    '''
    Ax = b with n iterations until error < e
    ''' 
    solutions = []    
    shape = np.shape(A)
    m = shape[0]
    x1 = x[:]
    D=np.diagonal(A)
    for i in range(m):
        x1[i] = (1-w)*x[i]+(w*(b[i]-np.dot(A[i,:i],x1[0:i])-np.dot(A[i,i+1:],x[i+1:]))/D[i])
        solutions.append(list(x1)) 
    x = x1
    n = 1
    err = 1
    while err > e:
        for i in range(m):
            x1[i] = (1-w)*x[i]+(w*(b[i]-np.dot(A[i,:i],x1[0:i])-np.dot(A[i,i+1:],x[i+1:]))/D[i])
            solutions.append(list(x1))             
        err = np.linalg.norm(x1-solutions[-2])/np.linalg.norm(x1)
        x = x1
        n += 1

    return n, solutions[-1]


def DE_solver(x_pos, y_pos, D_mesh, abs_mesh, source_mesh, err_tol):
    """
    x_pos: position of material along x direction (n+1 by 1)
    y_pos: position of material along y direction (m+1 by 1)
    D_mesh: Diffusion constant distribution over meshes (m by n)
    abs_mesh: Absorption macroscopic cross section distribution over meshes (m by n)
    source_mesh: Fixed source distribution over meshes (m by n)
    """
    m = len(y_pos)-1
    n = len(x_pos)-1
    coeff_matrix = np.zeros(((m+1)*(n+1),(m+1)*(n+1)))
    solution = np.empty(((m+1)*(n+1),1))
    solution.fill(np.nan)
    
    #Absorption: Non corner or edge points    
    abs_list = solution.copy()
    for j in reversed(range(1,m)): #the position starts from bottom left corner
        for i in range(1,n):
            d1=np.abs(x_pos[i]-x_pos[i-1])
            e1=np.abs(y_pos[j]-y_pos[j+1])
            d2=np.abs(x_pos[i+1]-x_pos[i])
            e2=np.abs(y_pos[j-1]-y_pos[j])

            V1=0.25*d1*e1
            V2=0.25*d2*e1
            V3=0.25*d2*e2
            V4=0.25*d1*e2
            
            abs_value = abs_mesh[j,i-1]*V1+abs_mesh[j,i]*V2+abs_mesh[j-1,i]*V3+abs_mesh[j-1,i-1]*V4
            source_value = source_mesh[j,i-1]*V1+source_mesh[j,i]*V2+source_mesh[j-1,i]*V3+source_mesh[j-1,i-1]*V4
            abs_list[len(y_pos)*len(x_pos)-(j+1)*len(x_pos)+(i+1)-1] = abs_value
            solution[len(y_pos)*len(x_pos)-(j+1)*len(x_pos)+(i+1)-1]= source_value
            
    #Absorption: Reflecting Side Right Points
    for j in reversed(range(1,m)): 
        i=n
        d1=np.abs(x_pos[i]-x_pos[i-1])
        e1=np.abs(y_pos[j]-y_pos[j+1])
        e2=np.abs(y_pos[j-1]-y_pos[j])
        
        V1=0.25*d1*e1
        V4=0.25*d1*e2

        abs_value = abs_mesh[j,i-1]*V1+abs_mesh[j-1,i-1]*V4
        source_value = source_mesh[j,i-1]*V1+source_mesh[j-1,i-1]*V4
        abs_list[len(y_pos)*len(x_pos)-(j+1)*len(x_pos)+(i+1)-1] = abs_value
        solution[len(y_pos)*len(x_pos)-(j+1)*len(x_pos)+(i+1)-1]= source_value       

    #Absorption: Reflecting Side top Points
    for i in range(1,n): 
        j=0
        d1=np.abs(x_pos[i]-x_pos[i-1])
        e1=np.abs(y_pos[j]-y_pos[j+1])
        d2=np.abs(x_pos[i+1]-x_pos[i])
        
        V1=0.25*d1*e1
        V2=0.25*d2*e1

        abs_value = abs_mesh[j,i-1]*V1+abs_mesh[j,i]*V2
        source_value = source_mesh[j,i-1]*V1+source_mesh[j,i]*V2
        abs_list[len(y_pos)*len(x_pos)-(j+1)*len(x_pos)+(i+1)-1] = abs_value
        solution[len(y_pos)*len(x_pos)-(j+1)*len(x_pos)+(i+1)-1]= source_value   
        
    #Absorption: Top Right
    i,j = n,0
    d1=np.abs(x_pos[i]-x_pos[i-1])
    e1=np.abs(y_pos[j]-y_pos[j+1])
    V1=0.25*d1*e1
    abs_value = abs_mesh[j,i-1]*V1
    source_value = source_mesh[j,i-1]*V1
    abs_list[len(y_pos)*len(x_pos)-(j+1)*len(x_pos)+(i+1)-1] = abs_value
    solution[len(y_pos)*len(x_pos)-(j+1)*len(x_pos)+(i+1)-1]= source_value
    
    
    Left=np.empty((n*(m+1),1))
    Left.fill(np.nan)
    Right=np.empty((n*(m+1),1))
    Right.fill(np.nan)
    Bottom=np.empty((m*(n+1),1))
    Bottom.fill(np.nan)
    Top=np.empty((m*(n+1),1))
    Top.fill(np.nan)
    Center=np.empty(((m+1)*(n+1),1))
    Center.fill(np.nan)
    
    #Flux: Non corner or edge points
    for j in reversed(range(1,m)): #the position starts from bottom left corner
        for i in range(1,n):
            d1=np.abs(x_pos[i]-x_pos[i-1])
            e1=np.abs(y_pos[j]-y_pos[j+1])
            d2=np.abs(x_pos[i+1]-x_pos[i])
            e2=np.abs(y_pos[j-1]-y_pos[j])
            
            a_L=-(D_mesh[j,i-1]*e1+D_mesh[j-1,i-1]*e2)/(2*d1)
            a_R=-(D_mesh[j,i]*e1+D_mesh[j-1,i]*e2)/(2*d2)
            a_B=-(D_mesh[j,i-1]*d1+D_mesh[j,i]*d2)/(2*e1)
            a_T=-(D_mesh[j-1,i-1]*d1+D_mesh[j-1,i]*d2)/(2*e2)
            a_C=abs_list[len(y_pos)*len(x_pos)-(j+1)*len(x_pos)+(i+1)-1]-(a_L+a_R+a_B+a_T)
            
            Left[((m+1-(j+1))*(n+1-1)+i-1)]=a_L
            Right[((m+1-(j+1))*(n+1-1)+i)]=a_R
            Bottom[(m+1-(j+1)-1)*(n+1)+i]=a_B
            Top[(m+1-(j+1))*(n+1)+i]=a_T
            Center[(m+1-(j+1))*(n+1)+i]=a_C
    
    #Flux: Reflecting side right
    for j in reversed(range(1,m)): #the position starts from bottom left corner
        i=n
        d1=np.abs(x_pos[i]-x_pos[i-1])
        e1=np.abs(y_pos[j]-y_pos[j+1])
        e2=np.abs(y_pos[j-1]-y_pos[j])

        a_L=-(D_mesh[j,i-1]*e1+D_mesh[j-1,i-1]*e2)/(2*d1)
        a_B=-(D_mesh[j,i-1]*d1)/(2*e1)
        a_T=-(D_mesh[j-1,i-1]*d1)/(2*e2)
        a_C=abs_list[len(y_pos)*len(x_pos)-(j+1)*len(x_pos)+(i+1)-1]-(a_L+a_B+a_T)

        Left[((m+1-(j+1))*(n+1-1)+i-1)]=a_L
        Bottom[(m+1-(j+1)-1)*(n+1)+i]=a_B
        Top[(m+1-(j+1))*(n+1)+i]=a_T
        Center[(m+1-(j+1))*(n+1)+i]=a_C 
        
    #Flux: Reflecting side top
    for i in range(1,n): #the position starts from bottom left corner
        j=0
        d1=np.abs(x_pos[i]-x_pos[i-1])
        e1=np.abs(y_pos[j]-y_pos[j+1])
        d2=np.abs(x_pos[i+1]-x_pos[i])
        #e2=np.abs(y_pos[j-1]-y[j])

        a_L=-(D_mesh[j,i-1]*e1)/(2*d1)
        a_R=-(D_mesh[j,i]*e1)/(2*d2)
        a_B=-(D_mesh[j,i-1]*d1+D_mesh[j,i]*d2)/(2*e1)
        a_C=abs_list[len(y_pos)*len(x_pos)-(j+1)*len(x_pos)+(i+1)-1]-(a_L+a_R+a_B)

        Left[(m+1-(j+1))*(n+1-1)+i-1]=a_L
        Right[((m+1-(j+1))*(n+1-1)+i)]=a_R
        Bottom[(m+1-(j+1)-1)*(n+1)+i]=a_B
        Center[(m+1-(j+1))*(n+1)+i]=a_C   
        
    #Flux: Top right
    i,j = n,0
    d1=np.abs(x_pos[i]-x_pos[i-1])
    e1=np.abs(y_pos[j]-y_pos[j+1])


    a_L=-(D_mesh[j,i-1]*e1)/(2*d1)
    a_B=-(D_mesh[j,i-1]*d1)/(2*e1)
    a_C=abs_list[len(y_pos)*len(x_pos)-(j+1)*len(x_pos)+(i+1)-1]-(a_L+a_B)

    Left[((m+1-(j+1))*(n+1-1)+i-1)]=a_L
    Bottom[(m+1-(j+1)-1)*(n+1)+i]=a_B
    Center[(m+1-(j+1))*(n+1)+i]=a_C

    for i in range((m+1)*(n+1)):
        coeff_matrix[i,i]=Center[i]
    
    for i in range(len(Top)):
        coeff_matrix[i,i+(n+1)*(m+1)-len(Top)]=Top[i]
        coeff_matrix[i+(n+1)*(m+1)-len(Bottom),i]=Bottom[i]

    A_Index=0
    V_Index=0
    for i in range((m+1)*(n+1)):
        skip = (i+1)%(n+1)
        if skip!=0: 
            coeff_matrix[A_Index,A_Index+1]=Right[V_Index]
            coeff_matrix[A_Index+1,A_Index]=Left[V_Index]
            A_Index+=1
            V_Index+=1
        else:
            A_Index+=1

    for i in range(len(solution)):
        if np.isnan(solution[i]):
            solution[i]=0
            coeff_matrix[i,:]= np.zeros((1,len(coeff_matrix[i,:])))
            coeff_matrix[i,i]=1
    guess = np.ones(((m+1)*(n+1),1))/np.linalg.norm(np.ones(((m+1)*(n+1),1)),2)

    iterations_gs, flux_list_gs = it_gauss_sedidel(np.array(coeff_matrix), np.array(solution).flatten(), np.array(guess).flatten() , err_tol)
    iterations_j, flux_list_j = it_jacobi(np.array(coeff_matrix),np.array(solution).flatten(), np.array(guess).flatten() , err_tol)
    iterations_s, flux_list_s = it_sor(np.array(coeff_matrix),np.array(solution).flatten(), np.array(guess).flatten() , err_tol)

    np.savetxt("gs_fulx.csv", flux_list_gs, delimiter=",")
    np.savetxt("jacobi_fulx.csv", flux_list_j, delimiter=",")
    np.savetxt("sor_fulx.csv", flux_list_s, delimiter=",")
    
    return flux_list_gs, flux_list_j, flux_list_s